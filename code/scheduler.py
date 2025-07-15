import schedule
import time
import os

def execute_file():
    start = time.time()  # 시작 시간 기록
    os.system('python s01.py')
    end = time.time()    # 종료 시간 기록
    elapsed = end - start
    print(f"한 번 도는 데 걸린 시간: {elapsed:.2f}초")

# 10분마다 파일 실행 스케줄 설정 (예시: 1초마다 실행)
schedule.every(1).seconds.do(execute_file)

# 스케줄을 유지하기 위해 스크립트를 계속 실행
while True:
    schedule.run_pending()
    time.sleep(1)
