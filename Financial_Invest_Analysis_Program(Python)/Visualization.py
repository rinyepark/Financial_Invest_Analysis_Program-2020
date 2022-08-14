import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd

from plotly.subplots import make_subplots


class Visualization:
    
    '''
    Description: 백테스트 결과로 모델별 시각화 및 벤치마크와의 비교
     (Constructor: param)
      - df: 벤치마크와 백테스트 결과 종합 데이터프레임
      - method_lst: 비교하고픈 최적화 모델의 목록
      - asset_name_lst: 사용한 자산의 이름 목록
     (Constructor: default param)
      - df_dd: drawdown을 시각화하기 위해 사용되는 데이터프레임(Initial제거)
      - amount: balance 계산 시 초기 투자금액
      
    Method
     (Public)
      - viz_pfo_return_with_market: 백테스트 결과와 벤치마크의 수익률 변동 추이 그래프 시각화
      - viz_pfo_balance_with_market: 백테스트 결과와 벤치마크의 투자금액 변화 추이 그래프 시각화
      - viz_pfo_balance_with_market_log: 투자금액 변화 추이 로그 스케일 그래프 시각화
      - viz_pfo_dd_with_market: 백테스트 결과와 벤치마크의 drawdown 그래프
      - viz_all: 모든 그래프 화면에 출력
     (Private)
     - _error_check: 에러 확인
     
    '''

    def __init__(self, df, method_lst, asset_name_lst):       
        self.df = df
        self.df_dd = df.iloc[1:]
        self.amount = df['Market_balance'].iloc[0]
        self.method_lst = method_lst
        self.asset_name_lst = asset_name_lst
        
        if not self._error_check():
            print("다시 입력하세요.")
        
        
    def _error_check(self):
        if not isinstance(self.method_lst,list) or not isinstance(self.asset_name_lst,list):
            print("[ERROR] list 형식으로 입력하세요")
            return False
        
        for i in self.asset_name_lst:
            for k in self.method_lst:
                for j in ["return","balance"]:
                    col_name= '{}_{}_{}'.format(i,k,j)
                    if col_name not in list(self.df.columns):
                        print("[ERROR] \'{}\'자산 백태스트 결과에 \'{}\' 모델의 결과가 없습니다.".format(i,k))
                        return False
        
        return True       

    def viz_pfo_return_with_market(self):
        if not self._error_check():
            print("다시 입력하세요.")
            return -1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df.Market_return,
                            mode='lines',
                            name='Market return'))
        
        for name in self.asset_name_lst:
            for i in self.method_lst:
                col_name = '{}_{}'.format(name,i)
                fig.add_trace(go.Scatter(x=self.df.index, y=self.df["{}_return".format(col_name)],
                                    mode='lines',
                                    name='{} {} pfo return'.format(name, i)))
            
        fig.update_layout(
            title="Portfolio growth with Market(return)",
        #     width=1000,
        #     height=400,
            xaxis = dict(title_text="date",),
            yaxis=dict(
                title_text="return",
            ),
            hovermode = 'x'
        )
        fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=len(self.df),
                y1=0,
                # opacity=0.5,
                line=dict(color='black',width=1)
        )
        fig.update_xaxes(type='category')
        fig.show()

    def viz_pfo_balance_with_market(self):
        if not self._error_check():
            print("다시 입력하세요.")
            return -1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df.Market_balance,
                            mode='lines',
                            name='Market balance'))
        for asset_name in self.asset_name_lst:
            for i in self.method_lst:
                col_name = '{}_{}'.format(asset_name, i)
                fig.add_trace(go.Scatter(x=self.df.index, y=self.df["{}_balance".format(col_name)],
                                    mode='lines',
                                    name='{} {} pfo balance'.format(asset_name, i)))

        fig.add_shape(type='line',
                x0=0,
                y0=self.amount,
                x1=len(self.df),
                y1=self.amount,
                # opacity=0.5,
                line=dict(color='black', width=1)
        )
        fig.update_layout(
            title="Portfolio growth with Market(balance)",
        #     width=1000,
        #     height=400,
            xaxis = dict(title_text="date",),
            yaxis=dict(
                title_text="balance",
            ),
            hovermode = 'x'
        )
        fig.update_xaxes(type='category')

        fig.show()
        
    def viz_pfo_balance_with_market_log(self):
        if not self._error_check():
            print("다시 입력하세요.")
            return -1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df.Market_balance,
                            mode='lines',
                            name='Market balance'))

        for asset_name in self.asset_name_lst:
            for i in self.method_lst:
                fig.add_trace(go.Scatter(x=self.df.index, y=self.df["{}_{}_balance".format(asset_name, i)],
                                    mode='lines',
                                    name='{} {} pfo balance'.format(asset_name, i)))
        fig.add_shape(type='line',
                x0=0,
                y0=self.amount,
                x1=len(self.df),
                y1=self.amount,
                # opacity=0.5,
                line=dict(color='black', width=1)
        )
        fig.update_layout(
            title="Portfolio growth with Market(balance)-log sclae",
        #     width=1000,
        #     height=400,
            xaxis = dict(title_text="date",),
            yaxis=dict(
                title_text="balance(log)",
            ),
            hovermode = 'x',
            yaxis_type="log"
        )
        fig.update_xaxes(type='category')

        fig.show()

        # DD 
    def viz_pfo_dd_with_market(self):
        if not self._error_check():
            print("다시 입력하세요.")
            return -1
        
        import warnings
        warnings.filterwarnings(action = 'ignore')      
        fig = go.Figure()
        window = 1

        Roll_Max = self.df_dd['Market_balance'].cummax()
        Daily_Drawdown = self.df_dd['Market_balance']/Roll_Max - 1.0
        self.df_dd['Market_DD'] = Daily_Drawdown

        fig.add_trace(go.Scatter(x=self.df_dd.index, y=self.df_dd.Market_DD,
                          mode='lines',
                          name='Market return'))
        for asset_name in self.asset_name_lst:
            for i in self.method_lst:
                Roll_Max = self.df_dd['{}_{}_balance'.format(asset_name,i)].cummax()
                Daily_Drawdown = self.df_dd['{}_{}_balance'.format(asset_name,i)]/Roll_Max - 1.0
                self.df_dd['{}_{}_DD'.format(asset_name,i)] = Daily_Drawdown
                fig.add_trace(go.Scatter(x=self.df_dd.index, y=self.df_dd["{}_{}_DD".format(asset_name,i)],
                                  mode='lines',
                                  name='{} {} pfo return'.format(asset_name,i)))

        fig.update_layout(
          title="Portfolio drawdown with Market(return)",
        #     width=1000,
        #     height=400,
          xaxis = dict(title_text="date",),
          yaxis=dict(
              title_text="return",
          ),
          hovermode = 'x'
        )

        fig.show()
        warnings.filterwarnings(action = 'default')
        
    def viz_all(self):
        if not self._error_check():
            print("다시 입력하세요.")
            return -1
        self.viz_pfo_return_with_market()
        self.viz_pfo_balance_with_market()
        self.viz_pfo_balance_with_market_log()
        self.viz_pfo_dd_with_market()