import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account

# 설정 (본인 환경에 맞게 수정)
EXCEL_FILE_PATH = r'C:\Users\Admin\OneDrive\바탕 화면\vscode\lgbm_predictions.csv'  # 실제 엑셀(CSV) 파일 경로
SHEET_ID = '1xjwl9zaUwAkl3q_lqDOVYh2ZzwZWE9Kf-ZK_5aT39Ns'  # Google 스프레드시트 ID
SERVICE_ACCOUNT_FILE = r'C:\Users\Admin\Downloads\tonal-premise-457803-h6-a4dbc8a6495d.json'  # 서비스 계정 JSON 파일 경로
SHEET_NAME = 'lgbm_predictions_7372282c0e1b4cc6bc160cbe8977bafa'  # 덮어쓸 시트 이름 (예: '시트1')

def update_google_sheet(excel_file_path, sheet_id, service_account_file, sheet_name):
    """
    엑셀 파일을 읽어 Google 스프레드시트의 데이터를 업데이트합니다.

    Args:
        excel_file_path (str): 엑셀(CSV) 파일 경로
        sheet_id (str): Google 스프레드시트 ID
        service_account_file (str): 서비스 계정 JSON 파일 경로
        sheet_name (str): 데이터를 덮어쓸 시트 이름
    """

    try:
        # Google Sheets API 서비스 생성
        creds = service_account.Credentials.from_service_account_file(service_account_file)
        service = build('sheets', 'v4', credentials=creds)

        # CSV 파일 읽기
        df = pd.read_csv(excel_file_path)

        # NaN 값을 빈 문자열로 대체
        df = df.fillna('')

        # 데이터를 리스트 형태로 변환
        data = [df.columns.values.tolist()] + df.values.tolist()

        # 기존 데이터 삭제
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1:ZZ"  # 또는 다른 적절한 범위 지정
        ).execute()

        # 새로운 데이터 쓰기
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1",  # 시작 위치만 지정
            valueInputOption="USER_ENTERED",
            body={"values": data}
        ).execute()

        print("Google 스프레드시트 업데이트 완료!")

    except Exception as e:
        print(f"오류 발생: {e}")

#if __name__ == "__main__":
update_google_sheet(EXCEL_FILE_PATH, SHEET_ID, SERVICE_ACCOUNT_FILE, SHEET_NAME)