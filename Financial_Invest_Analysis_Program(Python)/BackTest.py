from Portfolio_Optimization import *


class BackTest:
    
    '''
    Description: 백테스트 분석
     (Constructor: param)
      - data: 포트폴리오 구성 자산 수익률
      - data_name: 자산 이름
      - ff_data: 팩터 모델
      - reopt_period: re-optimize 주기
      - rebal_period: rebalancing 주기
      - lookback_period: lookback 기간
      - inv_start: 과거 투자시작 날짜
      - inv_end: 과거 투자종료 날짜
      - amount: 투자금액
     (Constructor: default param)
      - method_dict: 최적화 모델 종류
      - res_df: 백테스트 진행 과정에서 구한 포트폴리오 수익률(모델별)
      - weight_df_dict: 백테스트 진행 과정에서 구한 포트폴리오 가중치(모델별) 
    
    Method
     (Public)
      - start_backtest: 사용자로부터 입력된 포트폴리오 최적화 모델을 활용한 백테스트 진행
     (Private)
     - _date_str_to_int: date(YYYY-MM)형식을 year, mon 분리해서 int로 return
     - _cal_date_with_month: date(YYYY-MM형식)와 mon(개월 수) 연산
     - _df_to_estimate: 최적화 문제를 풀기 위해 필요한 estimate 계산
     - _pfo_return: 자산 평균 수익률과 가중치를 이용한 포트폴리오 return 계산
     - _rebalance_pfo: 포트폴리오 리밸런싱 진행
     
    '''
    
    def __init__(self, data, data_name, ff_data, reopt_period, rebal_period, lookback_period, inv_start, inv_end, amount=1):
        """
        === Description ===

        * parameter *
        data: historical data / type: dataframe
        data_name: name of assets / type: str
        reopt_period: re-optimizing assets weights period (month) / type: int
        rebal_period: rebalancing(evaluation) period (month) / type: int
        lookback_period: lookback period (month) / type: int
        inv_start: "YYYY-MM" / type: str
        inv_end: "YYYY-MM" / type: str
        amount: 투자금액 / type: int
        """
        self.data = data
        self.data_name = data_name
        self.ff_data = ff_data
        self.reopt_period = reopt_period
        self.rebal_period = rebal_period
        self.lookback_period = lookback_period
        self.inv_start = inv_start
        self.inv_end = inv_end
        self.amount = amount

        self.method_dict = {"Lasso":0, "Lasso_n":1, "Ridge":2, "Equal_weight":3, 
                            "Mean_variance":4, "Risk_parity":5, "Factor1":6, "Factor2":7, "Factor3":8, "AllSeason":9 ,"GB":10}

        
        # result(dataframe)
        self.res_df = pd.DataFrame([])
        self.res_df.index.name = "Date"
        
        self.weight_df_dict = dict()
        
    # date(YYYY-MM)형식을 year, mon 분리해서 int로 return
    # output: y(year,int), m(month, int)
    def _date_str_to_int(self, date):
        y, m = date.split('-')
        y, m = int(y), int(m)
        return y,m


    # date(YYYY-MM형식)와 mon(개월 수) 연산
    # output: 연산 후 date(YYYY-MM형식)
    def _cal_date_with_month(self, date, mon, oper='-'):
        y, m = self._date_str_to_int(date)
        if oper == '-':

            if m-mon < 1:
                m= m-mon
                while m < 1:
                  y -= 1
                  m = m + 12
            else:
                m -= mon

        elif oper == '+':
            if m+mon > 12:
                m=m+mon
                while m > 12:
                  y += 1
                  m = m - 12
            else:
                m += mon

        return '{0:04d}-{1:02d}'.format(y,m)

    def _df_to_estimate(self, df):
        """
            === Description ===

        함수 설명: 최적화 문제를 풀기 위해 필요한 estimate를 계산

        * df: estimate 계산에 이용

        output: estimate list 

          est_lst[0]: mean lst ex) [0.5,0.3, ..., 0.7] 
          est_lst[1]: var lst ex) [...]
          est_lst[2]: cov matrix ex) [[...],[...],...]
          est_lst[3]: holding-period return 


        """
        n=4
        est_lst = [0 for i in range(n)]

        est_lst[0] = df.mean()
        est_lst[1] = df.var()
        est_lst[2] = df.cov()

        #lookback 기간동안 발생한 holding-period return
        s = {}
        for i in df.columns:
            val = df.get(i)
            hldprd_ret = 1.0
            for j in val : 
                j += 1   #ror -> total return 곱연산으로 기간의 수익을 가져오기 위함.
                hldprd_ret*=j
                s.update({i:hldprd_ret})

        est_lst[3] = pd.Series(s)
        est_lst[3] = est_lst[3] - 1 #total return -> ror

        return est_lst
    
    def _pfo_return(self, mean, weight_lst):
        """
        === Description ===
        포트폴리오 return 계산

        * parameter *
        #est_mean: 특정 기간동안 estimate한 자산들의 평균 수익률 /type:Series
        mean: 수익률 /type:Series
        weight_lst: (가정)최적화를 통해 구한 weight / type:Series

        output: pfo return / type: float

        """
        ptf_ret = mean @ weight_lst

        return ptf_ret
    
    def _rebalance_pfo(self, pfo_amount, weight_lst):
        """
        === Description ===
        포트폴리오 rebalancing

        * parameter *
        pfo_amount: 해당 달 수익률이 더해져 바뀐 산업군 별 porfolio amount
        weight_lst: 각 산업군에 할당할 가중치
        """
        
        total = sum(pfo_amount)
        res_rebal = total * weight_lst

        return res_rebal

    def start_backtest(self, method, k=1, gamma=1, user_input = 10):
        print("Method:", method)
        
        tmp_res = pd.DataFrame([], columns = ["{}_{}_return".format(self.data_name, method),"{}_{}_balance".format(self.data_name, method)])
        tmp_end = self._cal_date_with_month(self.inv_end, 1,'+')
        tmp_date = self.inv_start
        
        weight_df = pd.DataFrame([], columns = self.data.columns)
        weight_df.index.name = "Date"
        
        # weight_lst 초기화
        weight_dict = {}
        for i in self.data.columns:
            weight_dict.update({i:0.1})
        weight_lst = pd.Series(weight_dict)

        # rebalance chcker # (12/12) 사용자 입력 rebalance 기간으로 리벨런스하는 작업진행.
        # reopt_chk >= self.reopt_period 가 참 일때(if) -> re-optimize 진행, 진행후 reopt_chk = 1
        # reopt_chk >= self.reopt_period 가 거짓 일때(else) -> reopt_chk++       
        reopt_chk = 1 #
        opt_flag = True # 최초 투자시 weight 분배가 필요하므로 flag로 최초1회 optimize 함.
        
        # rebal_chk >= self.rebal_period 가 참 일때(if) -> 리벨런스 진행, 리벨런스 진행후 rebal_chk = 1
        # rebal_chk >= self.rebal_period 가 거짓 일때(else) -> rebal_chk++       
        rebal_chk = 1 # 최초투자시점 부터 리벨런스가 필요하지 않으므로 1로 시작함.



        # inital_amount 가정
        size = len(self.data.columns)
        pfo_amount = np.array([1.0/size]*size)*self.amount


        
        tmp_res.loc["Initial"] = [0, self.amount] #res_df 초기값

        while tmp_date != tmp_end:
            # re-optimize 주기 확인후 re-optimize
            if reopt_chk >= self.reopt_period or opt_flag :
                
                # lookback 기간을 scope 하는 작업
                look_start = self._cal_date_with_month(tmp_date, self.lookback_period,'-')
                lookback_df = self.data.loc[look_start:tmp_date][:-1]

                # lookback_df를 이용한 estimate 계산 함수
                est_lst = self._df_to_estimate(lookback_df) #est_lst <- mean(산술평균) , var, cov

                # 구한 estimate를 이용해 weight를 구하는 함수(최적화)
                pfo_opt = Portfolio_Optimization(est_lst, self.data.loc[tmp_date])

                sel_num = self.method_dict[method]
                if sel_num == 0: # Lasso
                    weight_lst = pfo_opt.Lasso(k)
                elif sel_num == 1: #Lasso_n
                    print(look_start, tmp_date)
                    weight_lst = pfo_opt.Lasso_n(user_input, len(self.data.columns))
                elif sel_num == 2: #Ridge
                    weight_lst = pfo_opt.Ridge(gamma)
                elif sel_num == 3: #EQW
                    weight_lst = pfo_opt.Equal_weight()
                elif sel_num == 4: #MV
                    weight_lst = pfo_opt.Mean_variance()
                elif sel_num == 5: #RP
                    weight_lst = pfo_opt.Risk_parity()
                elif sel_num == 6: #F1
                    weight_lst = pfo_opt.Factor1(lookback_df, self.ff_data.loc[look_start:tmp_date][:-1])
                elif sel_num == 7: #F2
                    weight_lst = pfo_opt.Factor2(lookback_df, self.ff_data.loc[look_start:tmp_date][:-1])
                elif sel_num == 8: #F3
                    weight_lst = pfo_opt.Factor3(lookback_df, self.ff_data.loc[look_start:tmp_date][:-1])
                elif sel_num == 9: #AS
                    weight_lst = pfo_opt.AllSeason()
                elif sel_num == 10: #GB
                    weight_lst = pfo_opt.GB()
                else:
                    print("ERROR 0: 잘못된 method 입력입니다.(Equal_weight, Mean_variance, Risk_parity, Ridge, Lasso, Lasso_n)")
                    return 0


                reopt_chk = 1

                if opt_flag == True :
                    reopt_chk = 2
                    opt_flag = False

            else :
                reopt_chk = reopt_chk + 1
          
            #각 시점 자산별 weight를 어떻게 나눴는지 확인하기 위한 DataFrame 저장
            weight_df.loc[tmp_date] = np.array(weight_lst)

            pfo_amount *= (self.data.loc[tmp_date] + 1) #각 산업군 (amount * (1+수익률))
            tmp_res.loc[tmp_date] = [self._pfo_return(self.data.loc[tmp_date], weight_lst), sum(pfo_amount)]

            # 리벨런스 주기확인후 리벨런싱
            if rebal_chk >= self.rebal_period :
                pfo_amount = self._rebalance_pfo(pfo_amount, weight_lst)
                rebal_chk = 1
            else :
                rebal_chk = rebal_chk + 1   


            # 날짜 계산
            tmp_date = self._cal_date_with_month(tmp_date, 1, '+')

        
        self.res_df = pd.concat([self.res_df, tmp_res], axis = 1)
    
        
        self.weight_df_dict[method] = weight_df
        
        return self.res_df