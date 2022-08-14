import numpy as np
import pandas_datareader.data as web
from scipy import stats

import pandas as pd

from Portfolio_Optimization import *

class Simulation_MC:
    '''
    Description: Monte-Carlo simulation을 통한 분석
     (Constructor: default param)
      - asset_option: 자산 데이터 수익률 기준(month) *현재 시뮬레이션은 month 기준으로만 작동합니다.
      - pfo_mean: portfolio 평균 수익률
      - pfo_vol: portfolio volatility
      - simul_df: simulation 결과 저장 데이터프레임
      - fin_price: simulation 최종 금액 저장 데이터프레임
      - quan_table: 분위별 기준 값 저장
      - out_except_df: outlier를 제거한 simulation 최종 금액 저장 데이터프레임
      - MDD: 각 시뮬레이션 동안의 Max drawdown 값 저장 데이터프레임
      - investment: 투자금액
      
    Method
     (Public)
      - set_portfolio: 시뮬레이션에 이용할 자산 및 가중치 설정(자동/수동)
      - start_simulation: 시뮬레이션 진행
      - quantile_table: 분위별 기준 값 계산
      - viz_box_plot: quartile box plot 시각화
      - remove_outlier: outlier 제거
      - make_MDD: Max Drawdown 계산
      - viz_graph: End balance frequency histogram과 MDD 그래프 화면에 출력
    '''
    
    def __init__(self):
        self.asset_option = 0
        self.pfo_mean = 0
        self.pfo_vol = 0
        self.simul_df = pd.DataFrame()
        self.fin_price = pd.DataFrame()
        self.quan_table = pd.DataFrame([], columns = ["Quantile", "Price"])
        self.out_except_df = pd.DataFrame()
        self.MDD = pd.DataFrame()
        self.investment=0

    def set_portfolio(self, is_auto = True, asset_option = 'month', metric = 'Mean_variance', 
                  asset_select = 10, asset_ret = [], weight_lst = []):
        '''
        is_auto = True(default) -> 입력된 metric에 따라 자동으로 Portfolio Optimization을 통해 나온 가중치를 이용
                                   * metric
                                   * asset_select = 10(default) -> 사용할 특정 자산군 선택 가능
                  False         -> 사용자가 입력한 asset return과 weight lst를 이용해 포트폴리오 구성
                                   * asset_ret, weight_lst
        asset_option = 'month'(default) -> asset의 return 기간 기준을 함께 입력
        '''
        
        if is_auto:
            if len(asset_ret) == 0:
                if asset_option != 'month':
                    print("[ERROR -1]: 자산 기간 설정이 잘못되었습니다.(default asset 사용시 month 필수)")
                    return -1
                
                # asset_select대로(asset option 건드렸나 확인)
                else:
                    try:
                        tmp =  web.DataReader('{}_Industry_Portfolios'.format(asset_select), 'famafrench', start='1990-01-01', end='2020-09-01')
                    except:
                        print("[ERROR -1]: asset_select의 설정이 잘못되었습니다.(5,10,12,17,30,38,48,49 중 선택 가능)")
                        return -1
                    
                    asset_ret = tmp[0]/100
#             else:
#                 ret = asset_ret.copy()
            
            est_lst = [asset_ret.mean(), asset_ret.var(), asset_ret.cov()]
            
            self.asset_option = asset_option

            # Portfolio_Optimization 만들어서
            pfo_opt = Portfolio_Optimization(est_lst, asset_ret)
            if metric == 'Lasso':
                weight_lst = pfo_opt.Lasso()
            elif metric == 'Lasso_n':
                weight_lst = pfo_opt.Lasso_n()
            elif metric == 'Ridge':
                weight_lst = pfo_opt.Ridge()
            elif metric == 'Equal_weight': #EQW
                weight_lst = pfo_opt.Equal_weight()
            elif metric == 'Mean_variance': #MV
                weight_lst = pfo_opt.Mean_variance()
            elif metric == 'Risk_parity': #RP
                weight_lst = pfo_opt.Risk_parity()
            else:
                print("ERROR -1: 해당 최적화 모델을 존재하지 않습니다.")
                return -1
            
            self.pfo_mean = asset_ret.mean() @ weight_lst
            self.pfo_vol = np.sqrt(weight_lst.T.dot(asset_ret.cov()).dot(weight_lst))
        
        else:
            # 예외처리
            if len(weight_lst) == 0:
                print("[ERROR 0]: weight_lst의 입력이 없습니다.")
                return 0
            elif len(asset_ret) == 0:
                if asset_select != len(weight_lst):
                    print("[ERROR 0]: weight_lst의 개수가 default자산의 개수(10)와 일치하지 않습니다.")
                    return 0
                
                if asset_option != 'month':
                    print("[ERROR 0]: 자산 기간 설정이 잘못되었습니다.(default asset 사용시 month 필수)")
                    return 0
                tmp =  web.DataReader('{}_Industry_Portfolios'.format(asset_select), 'famafrench', start='1990-01-01', end='2020-09-01')
                asset_ret = tmp[0]/100 
            else:
                if len(weight_lst) != len(asset_ret.columns):
                    print("[ERROR 0]: 입력한 asset의 개수와 weight의 개수가 일치하지 않습니다.")
                    return 0
            
            # 포폴 mean, vol 구하기
            if asset_option in ['day', 'month', 'year']:
                self.asset_option = asset_option
                self.pfo_mean = asset_ret.mean() @ weight_lst
                self.pfo_vol = np.sqrt(weight_lst.T.dot(asset_ret.cov()).dot(weight_lst))
            else:
                print("[ERROR 1]: 잘못된 option입니다.")
                return 1
    
    def start_simulation(self, investment, duration,  
                         simul_num = 10000, sip = 0, v=10, option = 'month'):
        self.investment = investment
        self.simul_df = pd.DataFrame()
        self.fin_price = pd.DataFrame()
        rv = stats.t(v, self.pfo_mean, self.pfo_vol)
    
        for i in range(simul_num):
            price = investment
            
            tmp_price_lst = []
            for year in range(duration):
                new_ret = rv.rvs()
                price = price*(1+new_ret) + sip
                tmp_price_lst.append(price)
            self.simul_df[i] = tmp_price_lst
        
        self.fin_price = self.simul_df.iloc[-1,:].to_frame('Final_Price') #최종 값
        return self.fin_price.describe()
        
    def quantile_table(self, ran = [0.1, 0.25, 0.5, 0.75, 0.9]):
        self.quan_table = pd.DataFrame([], columns = ["Quantile", "Price"])

        for i in range(len(ran)):
            self.quan_table.loc[i] = ['{}%'.format(round(ran[i]*100,1)), fin_price.quantile(ran[i]).item()]
        return self.quan_table
    
    def viz_box_plot(self):
        if len(self.fin_price) == 0:
            print("'start_simulation'함수를 먼저 실행시켜주세요.")
            return 0
        fig = go.Figure()
        
        fig.add_trace(go.Box(x=self.fin_price['Final_Price']))

        fig.update_layout(title='Quartile',
                   xaxis_title='Month',
                   yaxis_title='Temperature (degrees F)')
        fig.show()

#     def viz_all_simulation(self):
#         if len(self.simul_df)==0:
#             print("'start_simulation'함수를 먼저 실행시켜주세요.")
#             return 0
    
#         fig = go.Figure()
#         for i in list(self.simul_df.columns):
#             fig.add_trace(go.Scatter(x=self.simul_df.index, y=self.simul_df[i],
#                                 mode='lines',
#                                 name=i))
#         fig.add_shape(type='line',
#                 x0=0,
#                 y0=self.investment,
#                 x1=max(list(self.simul_df.index)),
#                 y1=self.investment,
#                 line=dict(color='black',),
#                 xref='x',
#                 yref='y'
#                 )
#         fig.show()

    def remove_outlier(self):
        if len(self.fin_price) == 0:
            print("'start_simulation'함수를 먼저 실행시켜주세요.")
            return 0
        
        self.out_except_df = pd.DataFrame()
        
        IQR = self.fin_price.quantile(0.75) - self.fin_price.quantile(0.25)
        out_IQR = IQR * 1.5
        lowest = (self.fin_price.quantile(0.25) - out_IQR).item()
        highest = (self.fin_price.quantile(0.75) + out_IQR).item()
        self.out_except_df = self.fin_price[(self.fin_price["Final_Price"] >= lowest) & (self.fin_price["Final_Price"] <= highest)]
        
        return self.out_except_df
    
    def make_MDD(self):
        out_simul_df = self.simul_df[list(self.out_except_df.index)]
        out_simul_high = out_simul_df.cummax()
        mdd_lst = []

        for i in list(out_simul_df.columns):
            mdd_lst.append((out_simul_df[i]/out_simul_high[i] - 1.0).min())
        
        self.MDD = pd.DataFrame(mdd_lst, columns=['MDD'])
    
    def viz_graph(self):
        if len(self.out_except_df)==0:
            self.remove_outlier()
            
        if len(self.MDD)==0:
            self.make_MDD()

        percent = round(len(self.out_except_df)/len(self.fin_price) *100,2)
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Portfolio End Balance Histogram ({}%)'.format(percent), 
                                                            'Maximum Drawdown Histogram ({}%)'.format(percent)))

        fig.append_trace(go.Histogram(x=self.out_except_df['Final_Price'],
            name='Price', 
            marker_color='#EB89B5',
            opacity=0.75), row=1,col=1)

        fig.append_trace(go.Histogram(x=self.MDD['MDD'],
            name='MDD', 
            marker_color='#9966CC',
            opacity=0.75), row=2, col=1)

        fig.update_xaxes(title_text="End Balance($)", row=1, col=1)
        fig.update_xaxes(title_text="MAX Drawdown", row=2, col=1)

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        fig.update_layout(height=800, width=980,
                         bargap=0.01,
                         bargroupgap=0.1)

        fig.layout.xaxis2.tickformat=',.2%'

        fig.show()
        
