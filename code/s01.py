from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

print("ChromeDriver 시작 중...")
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--ignore-certificate-errors")  # SSL 인증서 오류 무시
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
options.add_argument("lang=ko-KR")  # 한국어 페이지 요청
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    print("웹페이지 접속 중: https://kr.investing.com/currencies/usd-krw")
    url = "https://kr.investing.com/currencies/usd-krw"
    driver.get(url)

    # 페이지 로드 상태 확인
    print("페이지 로드 상태 확인 중...")
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    print("페이지 로드 완료.")

    # XPath로 환율 요소 대기
    print("페이지 로드 대기 중 (최대 10초)...")
    wait = WebDriverWait(driver, 10)
    rate_element = wait.until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[2]/div[1]/div[1]/div[3]/div[1]/div[1]/div[1]'))
    )

    print("환율 데이터 추출 중...")
    rate = rate_element.text
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    data = {"usd_krw": rate, "timestamp": timestamp}
    with open("usd_krw_rate.json", "w") as f:
        json.dump(data, f)
    print("스크래핑 성공!")
    print(f"USD/KRW: {rate}")
    print(f"타임스탬프: {timestamp}")

except Exception as e:
    print("스크래핑 실패:", e)
    # 디버깅을 위해 페이지 소스 출력
    print("페이지 소스 (일부):")
    print(driver.page_source[:1000])  # 처음 1000자만 출력
finally:
    driver.quit()
    print("드라이버 종료.")