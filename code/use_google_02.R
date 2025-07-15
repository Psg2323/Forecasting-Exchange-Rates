library(shiny)
library(shinydashboard)
library(shinyDarkmode)
library(shinyjs)
library(plotly)
library(dplyr)
library(zoo)
library(RSelenium)
library(jsonlite)
library(googlesheets4)

# 작업 디렉토리 설정
setwd("C:/exchange/fianl")
message("초기 작업 디렉토리: ", getwd())

# UI 정의
ui <- dashboardPage(
  dashboardHeader(title = "USD/KRW 환율 대시보드"),
  dashboardSidebar(
    width = 300,
    div(
      style = "padding: 20px;",
      hr(),
      h4("차트 설정"),
      selectInput(
        "dateRangeSelect",
        "날짜 범위 선택",
        choices = list(
          "10일" = "10d",
          "1개월" = "1m",
          "3개월" = "3m",
          "6개월" = "6m",
          "9개월" = "9m",
          "1년" = "1y",
          "전체 데이터" = "all"
        ),
        selected = "6m"
      ),
      dateRangeInput("dateRange", "날짜 범위 선택",
                     start = Sys.Date() - 180,
                     end = Sys.Date() + 7),
      checkboxGroupInput(
        "display_options",
        "표시 옵션",
        choiceNames = list("실제 환율", "5일 MA", "20일 MA", "60일 MA", "120일 MA", "미래 예측"),
        choiceValues = list("actual", "ma5", "ma20", "ma60", "ma120", "future"),
        selected = c("actual", "future")
      ),
      actionButton("toggle_ma", "모든 이동평균 전환", icon = icon("chart-line")),
      actionButton("toggle_quarterly", "분기별 예측 전환", icon = icon("chart-area"), style = "margin-top: 10px; width: 100%;")
    )
  ),
  dashboardBody(
    use_darkmode(),
    useShinyjs(),
    tags$head(
      tags$style(HTML("
        .date-btn-group {
          display: flex;
          flex-wrap: wrap;
          justify-content: space-between;
          margin-bottom: 10px;
        }
        .btn-sm {
          flex: 1;
          margin: 0 2px;
          min-width: 60px;
          max-width: 80px;
          text-align: center;
        }
        #loading-overlay {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-color: rgba(0,0,0,0.5);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 9999;
        }
        .prediction-info {
          margin-top: 20px;
          padding: 15px;
          border-radius: 5px;
          background-color: var(--background-color);
          border: 1px solid var(--border-color);
          display: flex;
          justify-content: center;
          align-items: flex-start;
          flex-wrap: wrap;
          gap: 10px;
        }
        .prediction-value {
          font-weight: bold;
          color: #4CAF50;
        }
        .realtime-rate-container {
          text-align: center;
          margin: 0 10px;
          flex: 0 1 auto;
          min-width: 150px;
        }
        .today-future-container {
          display: flex;
          flex-direction: row;
          align-items: flex-start;
          gap: 10px;
        }
        .today-prediction-container {
          text-align: center;
          margin: 0;
          flex: 0 1 auto;
          min-width: 150px;
        }
        .future-predictions {
          text-align: center;
          margin: 0;
          flex: 0 1 auto;
          min-width: 200px;
        }
        .realtime-rate {
          font-size: 2em;
          font-weight: bold;
          color: #1E90FF;
        }
        .today-prediction {
          font-size: 2em;
          font-weight: bold;
          color: #FF0000;
        }
        .realtime-timestamp {
          font-size: 0.9em;
          color: #888;
        }
        .prediction-timestamp {
          font-size: 0.9em;
          color: #888;
        }
        .datepicker {
          z-index: 9999 !important;
        }
        #toggle_ma, #toggle_quarterly {
          margin-top: 10px;
          width: 100%;
        }
      "))
    ),
    div(id = "loading-overlay", style = "display: none;",
        div(
          style = "color: white; background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 5px;",
          h3("데이터 로딩 중..."),
          tags$div(class = "spinner-border", role = "status")
        )
    ),
    fluidRow(
      box(
        width = 12,
        plotlyOutput("main_plot", height = "600px"),
        div(class = "prediction-info",
            uiOutput("realtime_rate"),
            uiOutput("future_predictions")
        )
      )
    )
  )
)

# 서버 정의
server <- function(input, output, session) {
  darkmode(label = "🌓")
  loading_state <- reactiveVal(FALSE)
  show_quarterly <- reactiveVal(FALSE) # 분기별 예측 토글 상태
  
  # 데이터용 반응형 변수
  exchange_data <- reactiveVal(NULL)
  realtime_rate_data <- reactiveVal(NULL)
  quarterly_data <- reactiveVal(NULL)
  
  # 구글 시트 인증 및 분기별 데이터 로드 (앱 시작 시 1회)
  tryCatch({
    gs4_deauth()
    gs4_auth(cache = ".secrets", email = TRUE)
    message("구글 시트 인증 성공: ", Sys.time())
    
    # 분기별 데이터 스프레드시트 ID
    QUARTERLY_SHEET_ID <- "1zzhGXua3NoqHnlUJ5mJreMM7xEvncu8JPHNPOan8vC4"
    quarterly_df <- read_sheet(QUARTERLY_SHEET_ID, sheet = 1)
    
    # 데이터 처리
    required_cols <- c("date", "target_fx", "pac_fx_reer_opt")
    if (!all(required_cols %in% colnames(quarterly_df))) {
      stop("구글 시트에 필수 열(date, target_fx, pac_fx_reer_opt)이 없습니다.")
    }
    
    quarterly_df$date <- as.Date(quarterly_df$date, format = "%Y-%m-%d")
    quarterly_df <- quarterly_df %>%
      select(date, target_fx, pac_fx_reer_opt) %>%
      filter(!is.na(target_fx) | !is.na(pac_fx_reer_opt)) %>%
      arrange(date)
    
    quarterly_data(quarterly_df)
    message("분기별 데이터 로드 성공: ", nrow(quarterly_df), " 행, 날짜 범위: ", 
            min(quarterly_df$date, na.rm = TRUE), " ~ ", max(quarterly_df$date, na.rm = TRUE))
  }, error = function(e) {
    showNotification(paste("분기별 구글 시트 데이터 로드 오류:", e$message), type = "error")
    message("분기별 구글 시트 데이터 로드 오류: ", e$message)
  })
  
  # 기존 구글 시트 설정
  SHEET_ID <- "1xjwl9zaUwAkl3q_lqDOVYh2ZzwZWE9Kf-ZK_5aT39Ns"
  
  # JSON 파일 경로
  JSON_FILE_PATH <- normalizePath("C:/exchange/fianl/usd_krw_rate.json", mustWork = FALSE)
  
  # JSON 파일 로그
  message("JSON 파일 경로: ", JSON_FILE_PATH)
  message("JSON 파일 존재 여부: ", file.exists(JSON_FILE_PATH))
  if (file.exists(JSON_FILE_PATH)) {
    file_info <- file.info(JSON_FILE_PATH)
    message("JSON 파일 크기: ", file_info$size, " bytes")
    message("JSON 파일 권한: ", file_info$mode)
    raw_content <- tryCatch({
      readLines(JSON_FILE_PATH, warn = FALSE, encoding = "UTF-8")
    }, error = function(e) {
      message("JSON 파일 내용 읽기 오류: ", e$message)
      return(NULL)
    })
    if (!is.null(raw_content) && length(raw_content) > 0) {
      message("JSON 파일 내용 (첫 100자): ", 
              substr(paste(raw_content, collapse = ""), 1, 100))
    } else {
      message("JSON 파일이 비어 있거나 읽을 수 없습니다")
    }
  } else {
    message("디렉토리 내 파일 목록: ", list.files(dirname(JSON_FILE_PATH), pattern = "*.json"))
  }
  
  # 수동 JSON 로드 테스트
  message("수동 JSON 로드 테스트...")
  tryCatch({
    test_json <- jsonlite::fromJSON(JSON_FILE_PATH)
    message("수동 JSON 로드 성공: ", paste(capture.output(str(test_json)), collapse = "\n"))
  }, error = function(e) {
    message("수동 JSON 로드 실패: ", e$message)
  })
  
  # 기존 구글 시트 데이터 가져오기용 반응형 폴링
  sheet_data_trigger <- reactivePoll(
    intervalMillis = 10000,
    session = session,
    checkFunc = function() {
      message("구글 시트 데이터 확인: ", Sys.time())
      Sys.time()
    },
    valueFunc = function() {
      message("구글 시트 데이터 가져오기: ", Sys.time(), " - 작업 디렉토리: ", getwd())
      tryCatch({
        df <- read_sheet(SHEET_ID, sheet = 1)
        message("시트 데이터 가져오기 성공: ", Sys.time(), " - 행: ", nrow(df), 
                " 열: ", paste(colnames(df), collapse = ", "), 
                "\n샘플: ", paste(capture.output(head(df, 2)), collapse = "\n"))
        df
      }, error = function(e) {
        message("구글 시트 데이터 읽기 오류: ", e$message)
        showNotification(paste("구글 시트 읽기 실패:", e$message), type = "error")
        return(NULL)
      })
    }
  )
  
  # 실시간 환율 데이터용 반응형 타이머
  load_realtime_rate <- reactiveTimer(
    intervalMs = 5000,
    session = session
  )
  
  observe({
    load_realtime_rate()
    message("usd_krw_rate.json 로드 시도: ", Sys.time(), " - 작업 디렉토리: ", getwd())
    if (file.exists(JSON_FILE_PATH)) {
      tryCatch({
        raw_content <- readLines(JSON_FILE_PATH, warn = FALSE, encoding = "UTF-8")
        if (length(raw_content) == 0) {
          stop("JSON 파일이 비어 있습니다")
        }
        message("JSON 원본 내용: ", substr(paste(raw_content, collapse = ""), 1, 100))
        
        data <- fromJSON(JSON_FILE_PATH)
        message("usd_krw_rate.json 로드 성공: ", Sys.time(), " - 내용: ", 
                paste(capture.output(str(data)), collapse = "\n"))
        realtime_rate_data(data)
      }, error = function(e) {
        message("usd_krw_rate.json 로드 오류: ", e$message)
        showNotification(paste("usd_krw_rate.json 로드 오류:", e$message), type = "error")
        realtime_rate_data(NULL)
      })
    } else {
      message("usd_krw_rate.json 파일 없음: ", JSON_FILE_PATH, " 시간: ", Sys.time())
      showNotification("usd_krw_rate.json 파일을 찾을 수 없습니다.", type = "warning")
      realtime_rate_data(NULL)
    }
  })
  
  # 구글 시트 데이터 처리
  observe({
    df <- sheet_data_trigger()
    
    if (is.null(df) || nrow(df) == 0) {
      showNotification("구글 시트에서 데이터가 없습니다.", type = "error")
      return()
    }
    
    loading_state(TRUE)
    
    tryCatch({
      required_cols <- c("Date", "Actual", "Prediction")
      if (!all(required_cols %in% colnames(df))) {
        showNotification("구글 시트에 필수 열(Date, Actual, Prediction)이 없습니다.", type = "error")
        message("구글 시트에서 누락된 열: ", paste(setdiff(required_cols, colnames(df)), collapse = ", "))
        return()
      }
      
      colnames(df)[1] <- "date"
      df$date <- as.Date(df$date, format = "%Y-%m-%d")
      colnames(df)[which(colnames(df) == "Actual")] <- "Rate"
      colnames(df)[which(colnames(df) == "Predicted")] <- "Prediction"
      df <- df[order(df$date), ]
      
      df$MA5 <- rollmean(df$Rate, k = 5, fill = NA, align = "right")
      df$MA20 <- rollmean(df$Rate, k = 20, fill = NA, align = "right")
      df$MA60 <- rollmean(df$Rate, k = 60, fill = NA, align = "right")
      df$MA120 <- rollmean(df$Rate, k = 120, fill = NA, align = "right")
      
      exchange_data(df)
      message("exchange_data 업데이트 - 행: ", nrow(df), " 날짜 범위: ", min(df$date, na.rm = TRUE), " ~ ", max(df$date, na.rm = TRUE))
    }, error = function(e) {
      showNotification(paste("구글 시트 데이터 처리 오류:", e$message), type = "error")
      message("구글 시트 데이터 처리 오류: ", e$message)
    })
    
    loading_state(FALSE)
  })
  
  # 데이터 새로고침 로직
  refresh_data <- function() {
    if (loading_state()) return()
    
    loading_state(TRUE)
    
    tryCatch({
      df <- read_sheet(SHEET_ID, sheet = 1)
      
      required_cols <- c("Date", "Actual", "Prediction")
      if (!all(required_cols %in% colnames(df))) {
        showNotification("구글 시트에 필수 열(Date, Actual, Prediction)이 없습니다.", type = "error")
        message("구글 시트에서 누락된 열: ", paste(setdiff(required_cols, colnames(df)), collapse = ", "))
        return()
      }
      
      colnames(df)[1] <- "date"
      df$date <- as.Date(df$date, format = "%Y-%m-%d")
      colnames(df)[which(colnames(df) == "Actual")] <- "Rate"
      colnames(df)[which(colnames(df) == "Predicted")] <- "Prediction"
      df <- df[order(df$date), ]
      
      df$MA5 <- rollmean(df$Rate, k = 5, fill = NA, align = "right")
      df$MA20 <- rollmean(df$Rate, k = 20, fill = NA, align = "right")
      df$MA60 <- rollmean(df$Rate, k = 60, fill = NA, align = "right")
      df$MA120 <- rollmean(df$Rate, k = 120, fill = NA, align = "right")
      
      exchange_data(df)
      message("exchange_data 새로고침 - 행: ", nrow(df), " 날짜 범위: ", min(df$date, na.rm = TRUE), " ~ ", max(df$date, na.rm = TRUE))
    }, error = function(e) {
      showNotification(paste("구글 시트 데이터 새로고침 오류:", e$message), type = "error")
      message("구글 시트 데이터 새로고침 오류: ", e$message)
    })
    
    loading_state(FALSE)
  }
  
  observe({
    if (loading_state()) {
      shinyjs::show("loading-overlay")
    } else {
      shinyjs::hide("loading-overlay")
    }
  })
  
  observeEvent(input$toggle_ma, {
    current_options <- input$display_options
    ma_options <- c("ma5", "ma20", "ma60", "ma120")
    
    if (any(ma_options %in% current_options)) {
      new_options <- setdiff(current_options, ma_options)
      updateCheckboxGroupInput(session, "display_options", selected = new_options)
      shinyjs::runjs('document.getElementById("toggle_ma").innerText = "모든 이동평균 표시";')
    } else {
      new_options <- c(current_options, ma_options)
      updateCheckboxGroupInput(session, "display_options", selected = new_options)
      shinyjs::runjs('document.getElementById("toggle_ma").innerText = "모든 이동평균 숨기기";')
    }
    message("이동평균 토글 - 선택된 옵션: ", paste(new_options, collapse = ", "), " 시간: ", Sys.time())
  })
  
  observeEvent(input$toggle_quarterly, {
    current_state <- show_quarterly()
    show_quarterly(!current_state)
    
    if (show_quarterly()) {
      # 분기별 예측 모드: 전체 날짜 범위로 설정
      q_data <- quarterly_data()
      if (!is.null(q_data) && nrow(q_data) > 0) {
        min_date <- min(q_data$date, na.rm = TRUE)
        max_date <- max(q_data$date, na.rm = TRUE)
        updateDateRangeInput(session, "dateRange",
                             start = min_date,
                             end = max_date)
        message("분기별 예측 모드 - 날짜 범위 업데이트: ", min_date, " ~ ", max_date, " 시간: ", Sys.time())
      } else {
        showNotification("분기별 데이터가 없습니다.", type = "warning")
      }
      shinyjs::runjs('document.getElementById("toggle_quarterly").innerText = "일일 그래프 보기";')
    } else {
      # 일일 그래프 모드: 디폴트 6개월로 복원
      updateDateRangeInput(session, "dateRange",
                           start = Sys.Date() - 180,
                           end = Sys.Date() + 7)
      updateSelectInput(session, "dateRangeSelect", selected = "6m")
      message("일일 그래프 모드 - 날짜 범위 복원: ", Sys.Date() - 180, " ~ ", Sys.Date() + 7, " 시간: ", Sys.time())
      shinyjs::runjs('document.getElementById("toggle_quarterly").innerText = "분기별 예측 전환";')
    }
  })
  
  observeEvent(input$dateRangeSelect, {
    data <- exchange_data()
    if (is.null(data)) return()
    
    range <- switch(input$dateRangeSelect,
                    "10d" = list(start = Sys.Date() - 10, end = Sys.Date()),
                    "1m" = list(start = Sys.Date() - 30, end = Sys.Date()),
                    "3m" = list(start = Sys.Date() - 90, end = Sys.Date()),
                    "6m" = list(start = Sys.Date() - 180, end = Sys.Date()),
                    "9m" = list(start = Sys.Date() - 270, end = Sys.Date()),
                    "1y" = list(start = Sys.Date() - 365, end = Sys.Date()),
                    "all" = list(start = min(data$date, na.rm = TRUE), end = max(data$date, na.rm = TRUE)))
    
    updateDateRangeInput(session, "dateRange",
                         start = range$start,
                         end = range$end)
    message("날짜 범위 선택 업데이트 - 선택: ", input$dateRangeSelect, ", 범위: ", range$start, " ~ ", range$end, " 시간: ", Sys.time())
  })
  
  filtered_data <- reactive({
    data <- exchange_data()
    if (is.null(data) || nrow(data) == 0) {
      showNotification("선택한 날짜 범위에 대한 환율 데이터가 없습니다.", type = "warning")
      message("필터링된 데이터 없음: ", Sys.time())
      return(NULL)
    }
    
    data <- data %>%
      filter(date >= input$dateRange[1] & date <= input$dateRange[2])
    
    if (nrow(data) == 0) {
      showNotification("선택한 날짜 범위에 맞는 환율 데이터가 없습니다.", type = "warning")
      message("필터링된 데이터 비어 있음: ", input$dateRange[1], " ~ ", input$dateRange[2], " 시간: ", Sys.time())
      return(NULL)
    }
    
    message("필터링된 데이터 행: ", nrow(data), " 시간: ", Sys.time())
    data
  })
  
  filtered_prediction_data <- reactive({
    data <- exchange_data()
    if (is.null(data) || nrow(data) == 0) {
      showNotification("예측 데이터가 없습니다.", type = "warning")
      return(NULL)
    }
    
    data <- data %>%
      filter(date >= input$dateRange[1] & date <= input$dateRange[2] & !is.na(Prediction))
    
    if (nrow(data) == 0) {
      showNotification("선택한 날짜 범위에 맞는 예측 데이터가 없습니다.", type = "warning")
      return(NULL)
    }
    
    message("필터링된 예측 데이터 행: ", nrow(data), " 시간: ", Sys.time())
    data
  })
  
  future_prediction_data <- reactive({
    data <- exchange_data()
    if (is.null(data) || nrow(data) == 0) {
      showNotification("미래 예측 데이터가 없습니다.", type = "warning")
      return(NULL)
    }
    
    data <- data %>%
      filter(date >= Sys.Date() & !is.na(Prediction))
    
    if (nrow(data) == 0) {
      showNotification("오늘 이후의 미래 예측 데이터가 없습니다.", type = "warning")
      return(NULL)
    }
    
    message("미래 예측 데이터 행: ", nrow(data), " 시간: ", Sys.time())
    data
  })
  
  filtered_quarterly_data <- reactive({
    data <- quarterly_data()
    if (is.null(data) || nrow(data) == 0) {
      showNotification("분기별 데이터가 없습니다.", type = "warning")
      return(NULL)
    }
    
    data <- data %>%
      filter(date >= input$dateRange[1] & date <= input$dateRange[2])
    
    if (nrow(data) == 0) {
      showNotification("선택한 날짜 범위에 맞는 분기별 데이터가 없습니다.", type = "warning")
      return(NULL)
    }
    
    message("필터링된 분기별 데이터 행: ", nrow(data), " 시간: ", Sys.time())
    data
  })
  
  output$main_plot <- renderPlotly({
    if (show_quarterly()) {
      # 분기별 예측 그래프
      quarterly_data <- filtered_quarterly_data()
      if (is.null(quarterly_data)) {
        message("분기별 플로팅용 데이터 없음: ", Sys.time())
        return(NULL)
      }
      
      message("분기별 플롯 렌더링 - 행: ", nrow(quarterly_data), " 시간: ", Sys.time())
      
      p <- plot_ly() %>%
        add_trace(
          x = quarterly_data$date,
          y = quarterly_data$target_fx,
          type = "scatter",
          mode = "lines+markers",
          name = "실제 분기별 환율",
          line = list(color = "#00CED1", width = 1.5),
          marker = list(size = 4, color = "#00CED1")
        ) %>%
        add_trace(
          x = quarterly_data$date,
          y = quarterly_data$pac_fx_reer_opt,
          type = "scatter",
          mode = "lines+markers",
          name = "예측 분기별 환율",
          line = list(color = "#FF69B4", width = 1.5, dash = "dash"),
          marker = list(size = 4, color = "#FF69B4")
        ) %>%
        layout(
          xaxis = list(title = "날짜", tickangle = 45),
          yaxis = list(title = "USD/KRW 환율"),
          legend = list(
            orientation = "h",
            x = 0.5,
            xanchor = "center",
            y = 1.15,
            font = list(size = 12)
          ),
          hovermode = "x unified",
          margin = list(t = 100)
        )
      
      return(p)
    }
    
    # 기존 일일 그래프
    data <- filtered_data()
    if (is.null(data)) {
      message("일일 플로팅용 데이터 없음: ", Sys.time())
      return(NULL)
    }
    
    pred_data <- filtered_prediction_data()
    future_data <- future_prediction_data()
    
    message("일일 플롯 렌더링 - 행: ", nrow(data), " 시간: ", Sys.time())
    
    p <- plot_ly() %>%
      layout(
        xaxis = list(title = "날짜", tickangle = 45),
        yaxis = list(title = "USD/KRW 환율"),
        legend = list(
          orientation = "h",
          x = 0.5,
          xanchor = "center",
          y = 1.15,
          font = list(size = 12)
        ),
        hovermode = "x unified",
        margin = list(t = 100)
      )
    
    if("actual" %in% input$display_options) {
      p <- p %>% add_trace(
        x = data$date,
        y = data$Rate,
        type = "scatter",
        mode = "lines",
        name = "실제 환율",
        line = list(color = "#1E90FF", width = 2.5)
      )
    }
    
    if("ma5" %in% input$display_options) {
      p <- p %>% add_trace(
        x = data$date,
        y = data$MA5,
        type = "scatter",
        mode = "lines",
        name = "5일 이동평균",
        line = list(color = "#FF4500", width = 1.5, dash = "dash")
      )
    }
    
    if("ma20" %in% input$display_options) {
      p <- p %>% add_trace(
        x = data$date,
        y = data$MA20,
        type = "scatter",
        mode = "lines",
        name = "20일 이동평균",
        line = list(color = "#32CD32", width = 1.5, dash = "dot")
      )
    }
    
    if("ma60" %in% input$display_options) {
      p <- p %>% add_trace(
        x = data$date,
        y = data$MA60,
        type = "scatter",
        mode = "lines",
        name = "60일 이동평균",
        line = list(color = "#8A2BE2", width = "1.5", dash = "longdash")
      )
    }
    
    if("ma120" %in% input$display_options) {
      p <- p %>% add_trace(
        x = data$date,
        y = data$MA120,
        type = "scatter",
        mode = "lines",
        name = "120일 이동평균",
        line = list(color = "#FFD700", width = "1.5", dash = "dashdot")
      )
    }
    
    if("future" %in% input$display_options && !is.null(future_data)) {
      p <- p %>% add_trace(
        x = future_data$date,
        y = future_data$Prediction,
        type = "scatter",
        mode = "lines+markers",
        name = "미래 예측",
        line = list(color = "#FF0000", width = 1, dash = "dot"),
        marker = list(size = 6, color = "#FF0000")
      )
    }
    
    p
  })
  
  output$realtime_rate <- renderUI({
    rate_data <- realtime_rate_data()
    message("realtime_rate UI 렌더링: ", Sys.time(), " - 데이터 사용 가능: ", !is.null(rate_data))
    if (!is.null(rate_data) && "usd_krw" %in% names(rate_data) && "timestamp" %in% names(rate_data)) {
      tagList(
        div(class = "realtime-rate-container",
            h5("실시간 환율:"),
            div(class = "realtime-rate", sprintf("USD/KRW: %s", rate_data$usd_krw)),
            div(class = "realtime-timestamp", sprintf("업데이트: %s", rate_data$timestamp))
        )
      )
    } else {
      tagList(
        div(class = "realtime-rate-container",
            h5("실시간 환율:"),
            div(class = "realtime-rate", "실시간 환율 데이터를 불러올 수 없습니다.")
        )
      )
    }
  })
  
  output$future_predictions <- renderUI({
    future_data <- future_prediction_data()
    if (is.null(future_data)) {
      return(div(class = "future-predictions", "미래 예측 데이터가 없습니다."))
    }
    
    today_data <- future_data %>%
      filter(date == Sys.Date())
    
    future_only_data <- future_data %>%
      filter(date > Sys.Date())
    
    div(class = "today-future-container",
        div(class = "today-prediction-container",
            if (nrow(today_data) > 0) {
              div(
                h5("오늘 예측 환율:"),
                div(class = "today-prediction", sprintf("USD/KRW: %.2f", today_data$Prediction[1])),
                div(class = "prediction-timestamp", sprintf("날짜: %s", today_data$date[1]))
              )
            } else {
              div(
                h5("오늘 예측 환율:"),
                div(class = "today-prediction", "오늘 예측 데이터를 불러올 수 없습니다.")
              )
            }
        ),
        if (nrow(future_only_data) > 0) {
          div(class = "future-predictions",
              h5("미래 예측 환율:"),
              tags$ul(
                lapply(1:nrow(future_only_data), function(i) {
                  tags$li(sprintf("%s: %.2f KRW", future_only_data$date[i], future_only_data$Prediction[i]))
                })
              )
          )
        } else {
          div(class = "future-predictions", "미래 예측 데이터가 없습니다.")
        }
    )
  })
}

# 앱 실행
shinyApp(ui = ui, server = server)