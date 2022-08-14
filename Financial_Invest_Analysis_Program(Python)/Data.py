import pandas_datareader.data as web
import pandas as pd
import numpy as np

class Data:
    '''
    Description: 투자 분석에 사용할 자산 구성
     (Constructor: param)
      - name: 자산 이름 설정
      - isin_ind: 산업군 데이터 포함 여부
      - isin_add: 추가 데이터 포함 여부
      - ind_num: 산업군 데이터 종류
      - add_lst: 추가 데이터 선택 목록
    
    Method
     (Public)
      - prepare_data: 투자시작/종료 날짜 및 룩백기간을 고려한 최종 자산 구성
      - check_date: 투자시작/종료 날짜 및 룩백기간이 선택한 자산과 최소 투자가능날짜(1970-02)에 적합한지 확인
      - make_dataset: 선택한 데이터를 이용해 기간 고려하지 않고 자산 구성
     (Private)
      - _error_check: 에러 체크
      - _date_str_to_int: string형태의 날짜를 int로 변경
      - _cal_date_with_month: (특정 날짜) - (N개월) 계산
      - _update_min_date: 선택한 자산의 사용가능 기간을 고려해 최소 투자가능날짜 변경
      - _get_fama_french: fama-french 3 factor 데이터를 file로부터 읽어와 가공
    '''
    
    def __init__(self, name, isin_ind, isin_add=False, ind_num = 10, add_lst = []):
        self.name = name
        self.isin_ind = isin_ind
        self.isin_add = isin_add
        self.ind_num = ind_num
        self.add_lst = add_lst
        self.fin_df = pd.DataFrame([])
        self.bm_df = pd.DataFrame([])
        self.ff3_df = pd.DataFrame([])
        
        self.origin_df = pd.DataFrame([])
        self.origin_mb = pd.DataFrame([])
        self.origin_ff3 = pd.DataFrame([])
        
        self.add_asset_dict = {0:"US BOND 7-10", 1: "US BOND 20+", 2: "GOLD",
                              3: "COMMODITY" , 4:"S&P ETF"}
        self.limit_date_dict = {0:"2002-08", 1: "2002-08", 2: "2005-02",
                              3: "2006-03" , 4:"1993-02"}
        self.min_date = '1970-02'
        self.max_date = '2020-09'
        
        if not self._error_check():
            return 0
        
    def _error_check(self):
        if not self.isin_ind and not self.isin_add:
            print("[ERROR] 분석에 사용할 자산을 입력해주세요.")
            return False   
        if not isinstance(self.add_lst, list):
            print("[ERROR] 리스트 형태로 입력해주세요.")
            return False
        return True
    
    def _date_str_to_int(self, date):
        y, m = date.split('-')
        y, m = int(y), int(m)
        return y,m


    def _cal_date_with_month(self, date, mon):
        y, m = self._date_str_to_int(date)
        
        y -= mon // 12
        mon = mon % 12
        
        if m-mon < 1:
            y -= 1
            m = m + 12 - mon
        else:
            m -= mon

        return y,m

    def _update_min_date(self):
        if self.isin_add:
            for i in self.add_lst:
                new_date = self.limit_date_dict[i]
                
                y,m = self._date_str_to_int(self.min_date)
                y2, m2 = self._date_str_to_int(new_date)
                
                if y < y2:
                    self.min_date = new_date
                elif y == y2 and m < m2:
                    self.min_date = new_date
                    
    def prepare_data(self, inv_start,inv_end, lookback_period):
        self.make_dataset()
        
        y,m = self._cal_date_with_month(inv_start,lookback_period)
        look_start = '{0:04d}-{1:02d}'.format(y,m)
        
        self.fin_df = self.origin_df.loc[look_start:inv_end]
        self.bm_df = self.origin_bm.loc[look_start:inv_end]
        self.ff3_df = self.origin_ff3.loc[look_start:inv_end]
            
        
    
    def check_date(self, inv_start, inv_end, lookback_period):
        self._update_min_date()
        
        # inv_start와 룩백확인
        miny,minm = self._date_str_to_int(self.min_date)
        ly,lm = self._cal_date_with_month(inv_start,lookback_period)
        
        if ly < miny:
            print("[ERROR] 투자시작날짜와 lookback 기간이 제한된 시작날짜({})를 벗어납니다. 다시 설정해주세요".format(self.min_date))
            return False
        elif ly == miny and lm < minm:
            print("[ERROR] 투자시작날짜와 lookback 기간이 제한된 시작날짜({})를 벗어납니다. 다시 설정해주세요".format(self.min_date))
            return False
        
        # inv_end확인
        maxy,maxm = self._date_str_to_int(self.max_date)
        y2,m2 = self._date_str_to_int(inv_end)
        
        if y2 > maxy:
            print("[ERROR] 투자종료날짜가 제한된 종료날짜({})를 벗어납니다. 다시 설정해주세요".format(self.max_date))
            return False
        elif y2 == maxy and m2 > maxm:
            print("[ERROR] 투자종료날짜가 제한된 종료날짜({})를 벗어납니다. 다시 설정해주세요".format(self.max_date))
            return False
        
        # 컷팅해서 최종 데이터 줌
        return True
    
    def _get_fama_french(self):
        # Read the csv file again with skipped rows
        ff_factors = pd.read_csv('data/F-F_Research_Data_Factors.CSV', skiprows = 3, nrows = 1131, index_col = 0)

        # Format the date index
        ff_factors.index = pd.to_datetime(ff_factors.index, format= '%Y%m')

        # Convert from percent to decimal
        ff_factors = ff_factors.apply(lambda x: x/ 100)
        return ff_factors

    
    def make_dataset(self):
        total_asset_df = pd.DataFrame([])

        # asset
        if self.isin_ind:
            ind= web.DataReader('{}_Industry_Portfolios'.format(self.ind_num), 'famafrench', start=self.min_date, end='2020-09')
            ind_ret=ind[0]/100
            total_asset_df = pd.concat([total_asset_df, ind_ret], axis=1)

            
        if self.isin_add:
            # 10개 산업군 주식데이터는 위에서 다루었으니 주식관련 데이터를 제외하고
            # 다양한 투자 종목을 추가해 Risk parity, MVO의 전체적인 분산효과를 높혀보기 위하여
            # 여러 투자 자산을 추가 해봅니다.
            # ETF 운용사 선택 기준은 과거 데이터가 길게 남아 있는 것으로 우선하였습니다.

            # iShares 7-10 Year Treasury Bond ETF (IEF)       - 미국중기채       (2002.08 ~ )
            # iShares 20+ Year Treasury Bond ETF (TLT)        - 미국장기채       (2002.08 ~ )
            # iShares Gold Trust (IAU)                        - 금               (2005.02 ~ )
            # Invesco DB Commodity Index Tracking Fund (DBC)  - 원자재           (2006.03 ~ )
            # SPDR S&P 500 ETF Trust (SPY)                    - S&P 지수추종ETF  (1993.02 ~ )

            assets = web.get_data_yahoo(['IEF','TLT','IAU','DBC','SPY'], start='1989-12-01', end='2020-09-01',interval='m')['Adj Close'].pct_change()

            assets.rename(columns={'IEF':'US BOND 7-10'},inplace=True)
            assets.rename(columns={'TLT':'US BOND 20+'},inplace=True)
            assets.rename(columns={'IAU':'GOLD'},inplace=True)
            assets.rename(columns={'DBC':'COMMODITY'},inplace=True)
            assets.rename(columns={'SPY':'S&P ETF'},inplace=True)

            assets.index = assets.index.strftime("%Y-%m") 
            assets.index = pd.to_datetime(assets.index).to_period("M")
            
            new_asset = assets.iloc[:, self.add_lst]
            
            total_asset_df = pd.concat([total_asset_df, new_asset],axis=1)
            
        self.origin_df = total_asset_df.copy()
        
        # benchmark
        sp500 = web.get_data_yahoo('^GSPC', start='1970-02-01', end='2020-09-01',interval='m')
        sp500c = sp500['Adj Close'].pct_change().dropna()
        sp500c.index = sp500c.index.strftime("%Y-%m") 
        
        self.origin_bm = sp500c.copy()
        
        # Fama-French 3 Factor
        ff_data = self._get_fama_french()
        ff_data.index = ff_data.index.strftime("%Y-%m")
        ff_data = ff_data[self.min_date:'2020-09']
        
        self.origin_ff3 = ff_data.copy()
        
        
        
        
        
        