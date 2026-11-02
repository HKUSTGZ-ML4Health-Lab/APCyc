import argparse
import csv
import os
import json
import random
import re
import shutil
import threading
import time
from datetime import datetime, time as dtime
from itertools import cycle
from urllib.parse import urlparse
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.wait import TimeoutException as WaitTimeoutException
URL = 'https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolintrinsic'
CHECKPOINT = 'camsol_checkpoint.json'
RESULT_CSV = 'camsol_results_test.csv'
FAIL_CSV = 'camsol_failures_test.csv'
INPUT_FILE = None
TEST_COUNT = int(os.getenv('TEST_COUNT', '0') or '0')
LOGIN_EMAIL = os.environ.get('CAMSOL_USERNAME', '')
LOGIN_PASSWORD = os.environ.get('CAMSOL_PASSWORD', '')

def _pick_first_existing(*candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None
GECKODRIVER_PATH = _pick_first_existing(os.getenv('GECKODRIVER_PATH'), shutil.which('geckodriver'))
FIREFOX_BINARY = _pick_first_existing(os.getenv('FIREFOX_BINARY'), shutil.which('firefox'), shutil.which('firefox-esr'))
USE_WDM = os.getenv('USE_WDM', '0') == '1'
RUN_START = dtime(0, 0, 0)
RUN_END = dtime(23, 59, 59)
BASE_DELAY = (1, 2)
MICRO_DELAY = (0.2, 0.6)
BACKOFF_BASE = 30
BACKOFF_MAX = 1800
FORM_WAIT_SECONDS = 20
RESULT_WAIT_SECONDS = 120
NAV_RETRIES = 2
MAX_DRIVER_USES = 800
BLANK_PAGE_MIN_LEN = 400
SELF_HEAL_ON_FAIL = True
SEQ_MAX_RETRIES = 2
PROXY_POOL = []
PROXY_ROTATOR = cycle(PROXY_POOL) if PROXY_POOL else None
PROGRESS_BAR_WIDTH = 30
NUM_WORKERS = int(os.getenv('CAMSOL_WORKERS', '1'))

def log(msg: str) -> None:
    """统一日志格式，便于长时间运行时追踪。"""
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}')

def in_run_window() -> bool:
    """判断是否处于每天允许访问的时间窗口。"""
    now = datetime.now().time()
    if RUN_START < RUN_END:
        return RUN_START <= now <= RUN_END
    return now >= RUN_START or now <= RUN_END

def load_sequences(path: str):
    """
    读取 FASTA/纯文本序列，返回 (seq_id, sequence) 列表。
    如果缺少 FASTA 注释行，则自动生成 seq_编号。
    支持多行序列拼接。
    """
    seqs = []
    current_id = None
    buffer = []

    def flush():
        if buffer:
            seq = ''.join(buffer).strip()
            if seq:
                seq_id = current_id or f'seq_{len(seqs) + 1}'
                seqs.append((seq_id, seq))
        buffer.clear()
    with open(path) as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            if t.startswith('>'):
                flush()
                current_id = t[1:].strip() or f'seq_{len(seqs) + 1}'
            else:
                buffer.append(t)
    flush()
    return seqs

def load_checkpoint():
    if not CHECKPOINT:
        return {}
    return json.load(open(CHECKPOINT)) if os.path.exists(CHECKPOINT) else {}

def save_checkpoint(data):
    if not CHECKPOINT:
        return
    json.dump(data, open(CHECKPOINT, 'w'))

def ensure_result_csv():
    """确保结果文件存在并有表头。"""
    if not os.path.exists(RESULT_CSV):
        with open(RESULT_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(['id', 'sequence', 'camsol_score'])
    if not os.path.exists(FAIL_CSV):
        with open(FAIL_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(['id', 'sequence', 'error'])

def safe_sleep(a, b):
    """带随机的睡眠，模拟人为节奏。"""
    time.sleep(random.uniform(a, b))

def wait_for_window():
    """如果不在时间窗口内，就等待并定期检查。"""
    while not in_run_window():
        log('等待时间窗口打开中...')
        safe_sleep(240, 420)

def next_proxy():
    """如果配置了代理池，依次取下一个代理。格式建议 host:port 或 http://host:port。"""
    if PROXY_ROTATOR:
        return next(PROXY_ROTATOR)
    return None

def render_progress(done_count: int, total: int, start_ts: float) -> str:
    """
    构造一行进度条文本：百分比、条形、完成数/总数、ETA。
    不直接打印，便于在日志中复用。
    """
    percent = done_count / total if total else 1
    filled = int(PROGRESS_BAR_WIDTH * percent)
    bar = '#' * filled + '-' * (PROGRESS_BAR_WIDTH - filled)
    elapsed = time.time() - start_ts
    speed = elapsed / done_count if done_count else 0
    remaining = int(speed * (total - done_count)) if speed else 0
    eta_min, eta_sec = divmod(remaining, 60)
    return f'{percent * 100:5.1f}% [{bar}] {done_count}/{total} ETA {eta_min:02d}:{eta_sec:02d}'

def chunkify(seq_list, n):
    """将列表按顺序拆成 n 份，尽量平均。"""
    k, m = divmod(len(seq_list), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        chunks.append(seq_list[start:end])
        start = end
    return [c for c in chunks if c]

def build_driver(proxy: str=None):
    """创建新的 Firefox driver，可选代理。"""
    options = FirefoxOptions()
    options.add_argument('-headless')
    options.add_argument('--width=1280')
    options.add_argument('--height=720')
    if FIREFOX_BINARY:
        options.binary_location = FIREFOX_BINARY
    if proxy:
        parsed = urlparse(proxy if '://' in proxy else f'http://{proxy}')
        host, port = (parsed.hostname, parsed.port)
        if not host or not port:
            raise ValueError(f'代理格式无效：{proxy}，请使用 host:port 或 http://host:port')
        options.set_preference('network.proxy.type', 1)
        options.set_preference('network.proxy.http', host)
        options.set_preference('network.proxy.http_port', int(port))
        options.set_preference('network.proxy.ssl', host)
        options.set_preference('network.proxy.ssl_port', int(port))
        options.set_preference('network.proxy.no_proxies_on', '')
    if GECKODRIVER_PATH:
        service = FirefoxService(executable_path=GECKODRIVER_PATH)
    elif USE_WDM:
        service = FirefoxService(GeckoDriverManager().install())
    else:
        raise RuntimeError('未找到 geckodriver。请先安装 geckodriver 并设置 GECKODRIVER_PATH，或在有网络时设置 USE_WDM=1 允许自动下载。')
    driver = webdriver.Firefox(service=service, options=options)
    driver.set_page_load_timeout(120)
    driver.set_script_timeout(120)
    driver.implicitly_wait(1)
    return driver

def close_driver(driver):
    """安全关闭 driver。"""
    try:
        if driver:
            driver.quit()
    except Exception:
        pass

def wait_for_document_ready(driver, timeout=30):
    """等待页面 readyState=complete，避免半加载状态。"""
    WebDriverWait(driver, timeout).until(lambda d: d.execute_script('return document.readyState') == 'complete')

def switch_to_new_window(driver, old_handles, timeout=20):
    """等待新窗口/标签出现并切换过去。若未出现则保持当前窗口。"""
    try:
        WebDriverWait(driver, timeout, poll_frequency=0.5).until(lambda d: len(d.window_handles) > len(old_handles))
        new_handles = [h for h in driver.window_handles if h not in old_handles]
        if new_handles:
            driver.switch_to.window(new_handles[0])
            return True
    except WaitTimeoutException:
        return False
    return False

def is_access_denied(page_source: str) -> bool:
    """检测是否被拦截在未注册/未登录页面。"""
    if not page_source:
        return False
    lowered = page_source.lower()
    return 'access denied - user not recognized' in lowered or 'user_not_registered' in lowered

def perform_login(driver):
    """执行登录流程，成功后应处于已登录状态。"""
    log('检测到未登录，开始登录流程...')
    driver.get('https://www-cohsoftware.ch.cam.ac.uk/index.php/user_not_registered')
    wait_for_document_ready(driver, 30)
    try:
        login_link = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href,'login') or contains(translate(., 'LOGIN', 'login'), 'login')]")))
        login_link.click()
        wait_for_document_ready(driver, 20)
    except Exception:
        driver.get('https://www-cohsoftware.ch.cam.ac.uk/index.php/login')
        wait_for_document_ready(driver, 20)
    try:
        consent_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(translate(., 'ACCEPT', 'accept'),'accept') or contains(translate(.,'AGREE','agree'),'agree') or @data-cc='show-consentModal']")))
        consent_btn.click()
        safe_sleep(0.5, 1.0)
    except Exception:
        pass
    try:
        remember = driver.find_element(By.ID, 'uMaintainLogin')
        if remember.is_displayed() and (not remember.is_selected()):
            remember.click()
    except Exception:
        pass
    email_selectors = ["//input[@id='uName']", "//input[@id='uEmail']", "//input[@name='uName']", "//input[@type='email']", "//input[@type='text']", "//input[contains(@name,'mail')]", "//input[contains(@name,'name')]"]
    pwd_selectors = ["//input[@id='uPassword']", "//input[@name='uPassword']", "//input[@type='password']", "//input[contains(@name,'pass')]"]
    email_input = None
    for xpath in email_selectors:
        try:
            email_input = WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.XPATH, xpath)))
            break
        except Exception:
            continue
    if email_input is None:
        try:
            candidates = driver.find_elements(By.CSS_SELECTOR, 'input')
            for el in candidates:
                typ = (el.get_attribute('type') or '').lower()
                if typ in ('email', 'text', '') and el.is_displayed():
                    email_input = el
                    break
        except Exception:
            pass
    if email_input is None:
        raise TimeoutException('未找到邮箱/用户名输入框')
    pwd_input = None
    for xpath in pwd_selectors:
        try:
            pwd_input = WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.XPATH, xpath)))
            break
        except Exception:
            continue
    if pwd_input is None:
        try:
            candidates = driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
            for el in candidates:
                if el.is_displayed():
                    pwd_input = el
                    break
        except Exception:
            pass
    if pwd_input is None:
        raise TimeoutException('未找到密码输入框')
    email_input.clear()
    email_input.send_keys(LOGIN_EMAIL)
    safe_sleep(0.3, 0.7)
    pwd_input.clear()
    pwd_input.send_keys(LOGIN_PASSWORD)
    safe_sleep(0.3, 0.7)
    submit_selectors = ["//button[@type='submit']", "//button[contains(translate(., 'SIGN IN', 'sign in'),'sign in')]", "//button[contains(translate(., 'LOGIN', 'login'),'login')]", "//input[@type='submit']", "//input[contains(translate(@value,'LOGIN','login'),'login')]", "//input[contains(translate(@value,'SIGN','sign'),'sign')]"]
    submit_btn = None
    for xpath in submit_selectors:
        try:
            submit_btn = WebDriverWait(driver, 12).until(EC.element_to_be_clickable((By.XPATH, xpath)))
            break
        except Exception:
            continue
    if submit_btn is None:
        try:
            for el in driver.find_elements(By.TAG_NAME, 'button'):
                if el.is_displayed():
                    submit_btn = el
                    break
            if submit_btn is None:
                for el in driver.find_elements(By.CSS_SELECTOR, "input[type='submit']"):
                    if el.is_displayed():
                        submit_btn = el
                        break
        except Exception:
            pass
    if submit_btn is None:
        raise TimeoutException('未找到登录提交按钮')
    submit_btn.click()
    wait_for_document_ready(driver, 20)
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, "//a[contains(translate(., 'CAMSOL', 'camsol'),'camsol')]")))
    except Exception:
        pass
    log('登录流程完成，继续处理。')
    try:
        src = driver.page_source or ''
        if src:
            with open('login_debug.html', 'w', encoding='utf-8') as dbg:
                dbg.write(src)
    except Exception:
        pass

def login_and_capture_on_fail(driver):
    """包装登录流程，失败时保存页面源码便于诊断。"""
    try:
        perform_login(driver)
    except Exception as exc:
        try:
            src = driver.page_source or ''
            with open('login_fail.html', 'w', encoding='utf-8') as dbg:
                dbg.write(src)
            log(f'登录失败，已保存 login_fail.html，错误：{exc}')
        except Exception:
            log(f'登录失败且无法保存页面源码：{exc}')
        raise

def goto_camsol_after_login(driver):
    """
    按指定流程：先登录，再从主页点击“CamSol Intrinsic”进入交互页面。
    返回时应已在 CamSol 页面。
    """
    login_and_capture_on_fail(driver)
    safe_sleep(0.5, 1.0)
    camsol_link = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//a[contains(translate(., 'CAMSOL', 'camsol'),'camsol')]")))
    camsol_link.click()
    wait_for_document_ready(driver, 30)
    log('已从主页进入 CamSol 页面。')

def load_input_form(driver):
    """加载表单页，检测空白/部分加载并重试。"""
    last_exc = None
    for attempt in range(1, NAV_RETRIES + 1):
        try:
            driver.get(URL)
            wait_for_document_ready(driver, FORM_WAIT_SECONDS)
            if is_access_denied(driver.page_source or ''):
                goto_camsol_after_login(driver)
                safe_sleep(1, 2)
                driver.get(URL)
                wait_for_document_ready(driver, FORM_WAIT_SECONDS)
                if is_access_denied(driver.page_source or ''):
                    raise TimeoutException('登录后仍被拒绝访问')
                safe_sleep(1, 2)
                continue
            if len(driver.page_source or '') < BLANK_PAGE_MIN_LEN:
                raise TimeoutException('页面内容过短，疑似空白加载')
            textarea = None
            selectors = [(By.CSS_SELECTOR, 'textarea'), (By.CSS_SELECTOR, "textarea[name='protein']"), (By.TAG_NAME, 'textarea')]
            for by, sel in selectors:
                try:
                    textarea = WebDriverWait(driver, FORM_WAIT_SECONDS).until(EC.visibility_of_element_located((by, sel)))
                    break
                except Exception:
                    continue
            if textarea is None:
                raise TimeoutException('未找到输入文本框')
            return textarea
        except (TimeoutException, WebDriverException) as exc:
            last_exc = exc
            log(f'加载表单失败({attempt}/{NAV_RETRIES})：{exc}')
            try:
                src = driver.page_source or ''
                log(f'当前页面长度：{len(src)}')
                if attempt == NAV_RETRIES:
                    with open('camsol_debug.html', 'w', encoding='utf-8') as dbg:
                        dbg.write(src)
                    log('已将当前页面源码保存到 camsol_debug.html 供排查')
            except Exception:
                pass
            safe_sleep(5, 10)
    raise last_exc or TimeoutException('无法打开 CamSol 表单页')

def parse_score_from_source(src: str) -> float:
    """从页面源代码中解析得分。"""
    match = re.search('intrinsic\\s+solubility\\s+score\\s+is\\s+([+-]?\\d+\\.\\d+)', src, flags=re.IGNORECASE)
    if not match:
        raise ValueError('未在结果页找到得分字段')
    return float(match.group(1))

def wait_for_result_page(driver) -> float:
    """等待结果页加载完并解析得分。"""

    def _has_score(_):
        src = driver.page_source or ''
        if len(src) < BLANK_PAGE_MIN_LEN:
            return False
        return re.search('intrinsic\\s+solubility\\s+score\\s+is\\s+[+-]?\\d+\\.\\d+', src, flags=re.IGNORECASE)
    WebDriverWait(driver, RESULT_WAIT_SECONDS, poll_frequency=2).until(_has_score)
    wait_for_document_ready(driver, 10)
    return parse_score_from_source(driver.page_source)

def submit_sequence(driver, sequence: str) -> float:
    """
    在当前 driver 上提交单条序列并返回得分。
    包含对过早提交、未跳转、空白页的检测。
    """
    textarea = load_input_form(driver)
    textarea.clear()
    safe_sleep(*MICRO_DELAY)
    textarea.send_keys(sequence)
    safe_sleep(*MICRO_DELAY)
    old_handles = list(driver.window_handles)
    submit_btn = WebDriverWait(driver, FORM_WAIT_SECONDS).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit']")))
    submit_btn.click()
    switched = switch_to_new_window(driver, old_handles, timeout=20)
    if switched:
        wait_for_document_ready(driver, 30)
    return wait_for_result_page(driver)

def run_single_sequence(driver, sequence: str, max_attempts: int=3) -> float:
    """带重试的单条序列执行，针对短暂网络/页面错误自愈。"""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return submit_sequence(driver, sequence)
        except (TimeoutException, StaleElementReferenceException, WebDriverException, ValueError) as exc:
            last_exc = exc
            safe_sleep(8, 16)
            try:
                driver.refresh()
                wait_for_document_ready(driver, 10)
            except Exception:
                pass
    raise last_exc or RuntimeError('达到最大重试次数')

def main():
    global CHECKPOINT, RESULT_CSV, FAIL_CSV, INPUT_FILE
    parser = argparse.ArgumentParser(description='Run CamSol Intrinsic scoring with Selenium.')
    parser.add_argument('--input-file', required=True, help='FASTA/plain-text input file path.')
    parser.add_argument('--result-csv', default=RESULT_CSV, help='Output CSV path for successful results.')
    parser.add_argument('--fail-csv', default=FAIL_CSV, help='Output CSV path for failures.')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Checkpoint file path for resumable runs.')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpoint and always start fresh.')
    args = parser.parse_args()
    INPUT_FILE = args.input_file
    RESULT_CSV = args.result_csv
    FAIL_CSV = args.fail_csv
    CHECKPOINT = None if args.no_checkpoint else args.checkpoint
    if args.no_checkpoint:
        log('Checkpoint disabled: starting fresh run.')
    seqs = load_sequences(INPUT_FILE)
    done = load_checkpoint()
    ensure_result_csv()
    if TEST_COUNT > 0:
        seqs = seqs[:TEST_COUNT]
        log(f'TEST 模式：仅处理前 {TEST_COUNT} 条序列用于试跑验证')
    todo = [(sid, s) for sid, s in seqs if sid not in done]
    total = len(seqs)
    start_ts = time.time()
    log(f'总共 {total} 条，已完成 {len(done)} 条，待做 {len(todo)} 条')
    lock = threading.Lock()

    def worker(seq_chunk):
        nonlocal done, start_ts
        backoff = BACKOFF_BASE
        driver = None
        driver_use_count = 0
        proxy_in_use = None
        for seq_id, seq in seq_chunk:
            if len(seq) < 7:
                with lock:
                    log(f'[跳过] {seq_id} 序列长度小于 7，记为失败')
                    with open(FAIL_CSV, 'a', newline='') as f:
                        csv.writer(f).writerow([seq_id, seq, 'length < 7'])
                    done[seq_id] = None
                    save_checkpoint(done)
                continue
            attempt = 0
            success = False
            while attempt < SEQ_MAX_RETRIES and (not success):
                attempt += 1
                wait_for_window()
                if driver is None or driver_use_count >= MAX_DRIVER_USES:
                    close_driver(driver)
                    proxy_in_use = next_proxy()
                    if proxy_in_use:
                        log(f'切换代理：{proxy_in_use}')
                    log('启动新的浏览器实例...')
                    driver = build_driver(proxy=proxy_in_use)
                    goto_camsol_after_login(driver)
                    driver_use_count = 0
                    safe_sleep(2, 5)
                try:
                    log(f'[CamSol] 处理序列: {seq_id} | {seq[:20]}... (尝试 {attempt}/{SEQ_MAX_RETRIES})')
                    score = run_single_sequence(driver, seq)
                    log(f'  → score = {score}')
                    with lock:
                        done[seq_id] = score
                        save_checkpoint(done)
                        with open(RESULT_CSV, 'a', newline='') as f:
                            csv.writer(f).writerow([seq_id, seq, score])
                        progress_line = render_progress(len(done), total, start_ts)
                        print(progress_line, end='\r', flush=True)
                    driver_use_count += 1
                    safe_sleep(*BASE_DELAY)
                    backoff = BACKOFF_BASE
                    success = True
                except Exception as exc:
                    wait_time = random.uniform(backoff, backoff + 20)
                    safe_sleep(wait_time, wait_time + 1)
                    backoff = min(backoff * 2, BACKOFF_MAX)
                    if SELF_HEAL_ON_FAIL:
                        close_driver(driver)
                        driver = None
                        driver_use_count = 0
                    if attempt >= SEQ_MAX_RETRIES:
                        with lock:
                            with open(FAIL_CSV, 'a', newline='') as f:
                                csv.writer(f).writerow([seq_id, seq, str(exc)])
                            done[seq_id] = None
                            save_checkpoint(done)
        close_driver(driver)
    chunks = chunkify(todo, NUM_WORKERS)
    threads = []
    def guarded_worker(chunk):
        try:
            worker(chunk)
        except Exception as exc:
            # Browser startup/login can fail before per-sequence retries start.
            # Propagate that failure into the table and process exit status.
            with lock:
                for sid, sequence in chunk:
                    if sid not in done:
                        with open(FAIL_CSV, 'a', newline='') as f:
                            csv.writer(f).writerow([sid, sequence, f'{type(exc).__name__}: {exc}'])
                        done[sid] = None
                save_checkpoint(done)
            log(f'Browser worker failed: {type(exc).__name__}')
    for chunk in chunks:
        t = threading.Thread(target=guarded_worker, args=(chunk,), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    if total:
        print(render_progress(len(done), total, start_ts))
    if any(done.get(sid) is None for sid, _ in seqs):
        raise SystemExit('CamSol did not produce a score for every requested sequence; inspect failures CSV.')
if __name__ == '__main__':
    main()
