import numpy as np
import pandas as pd

class Performance_Evaluation:
    '''
    Description: 백테스트 결과로 모델별 성능 평가 및 벤치마크와의 비교
     (Constructor: param)
      - df: 벤치마크와 백테스트 결과 종합 데이터프레임
      - option: 데이터 수익률 기준(day,month year)
     (Constructor: default param)
      - size: 입력된 데이터프레임의 크기
      - compare_df: 성능평가 계산 결과
      
    Method
     (Public)
      - arithmetic_mean: arithmetic mean 계산
      - annualized_art: annualized arithmetic mean 계산
      - hold_period_ret: holding period return 계산
      - annualized_geo: annualized geometric mean 계산
      - variance: variance 계산
      - volatility: volatility 계산
      - annual_vol: annual volaility 계산
      - Sharpe_ratio: Sharpe ratio 계산
      - Sharpe_ratio_annual: annualized sharpe ratio 계산
      - covariance: Market과의 covariance 계산
      - Bata: Market과의 Beta 계산
      - Alpha: Market과의 Alpha 계산
      - VaR: Value at Risk 계산
      - CVaR: Conditional Value at Risk 계산
      - MDD: Max Drawdown 계산
      
      - compare_pfo_weigh_bm: 입력된 자산들의 백테스트 결과와 벤치마크의 평가척도를 계산해 비교
     (Private)
      - _error_check: 에러 확인
      - _clean_val: 계산 결과 값 반올림
     
    '''
    
    
    def __init__(self, df, option='month'):
        self.df= df.iloc[1:]
        self.size = len(df)
        self.option = option
        self.compare_df = pd.DataFrame([])
        
        
        
    def _error_check(self, method_lst, asset_name_lst):
        if not isinstance(method_lst,list) or not isinstance(asset_name_lst,list):
            print("[ERROR] list 형식으로 입력하세요")
            return False
        
        for i in asset_name_lst:
            for k in method_lst:
                for j in ["return","balance"]:
                    col_name= '{}_{}_{}'.format(i,k,j)
                    if col_name not in list(self.df.columns):
                        print("[ERROR] \'{}\'자산 백태스트 결과에 \'{}\' 모델의 결과가 없습니다.".format(i,k))
                        return False
        
        return True       

        
    def _clean_val(self, num, is_clean):
        if is_clean:
            return round(num,5)
        else:
            return num
        
    def arithmetic_mean(self, method, is_clean=True):
        col_name = "{}_return".format(method)
        mean = self.df[[col_name]].mean().item()
        
        return self._clean_val(mean, is_clean)
    
    def annualized_art(self, method, is_clean=True):
        """
        3장 - return conversion참조
        """

        tmp = 12 #month
        if self.option == 'week':
            tmp = 52
        elif self.option == 'day':
            tmp = 252
        
        mean = tmp * self.arithmetic_mean(method, False)
        
        return self._clean_val(mean, is_clean)
    
    def hold_period_ret(self, method, is_clean = True):
        col_name = "{}_balance".format(method)
        
        beg = self.df[[col_name]].iloc[0]
        end = self.df[[col_name]].iloc[-1]
        
        ret = (end/beg-1).item()

        return self._clean_val(ret,is_clean)
    

    def annualized_geo(self, method, is_clean=True):

        tmp = 12 #month
        if self.option == 'week':
            tmp = 52
        elif self.option == 'day':
            tmp = 252

        num_of_year = self.size/tmp
        h_period_ret = self.hold_period_ret(method, False)

        geo_num = (1+h_period_ret)**(1/num_of_year)-1
        
        return self._clean_val(geo_num, is_clean)
    
    def variance(self,method, is_clean=True):
        col_name = "{}_return".format(method)
        var = self.df[[col_name]].var().item()
        
        return self._clean_val(var, is_clean)
    
    def volatility(self, method, is_clean=True):
        var = self.variance(method, False)
        vol = np.sqrt(var).item()
        
        return self._clean_val(vol, is_clean)
    
    def annual_vol(self, method, is_clean=True):
        tmp = 12
        if self.option == 'week':
            tmp = 52
        elif self.option == 'day':
            tmp = 252

        vol = self.volatility(method, False)
        res = np.sqrt(tmp) * vol
        
        return self._clean_val(res,is_clean)

    def Sharpe_ratio(self, method, risk_free = 0.001, is_clean=True):
        mean = self.arithmetic_mean(method, False)
        vol = self.volatility(method, False)
        
        sr = (mean-risk_free)/vol
        return self._clean_val(sr, is_clean)
    
    def Sharpe_ratio_annual(self, method, risk_free = 0.001, is_clean=True):
        a_mean = self.annualized_art(method, False)
        a_vol = self.annual_vol(method, False)
        
        sra = (a_mean-12*risk_free)/a_vol
        return self._clean_val(sra,is_clean)
    
    def covariance(self, method, is_clean = True):
        if method == "Market":
            print("*ERROR: method는 Market이 될 수 없습니다.")
            return np.nan

        col_name= "{}_return".format(method)
        covar = self.df[[col_name, "Market_return"]].cov().iloc[0,1]

        return self._clean_val(covar, is_clean)
        
    def Beta(self, method, is_clean=True):
        if method == "Market":
            print("*ERROR: method는 Market이 될 수 없습니다.")
            return np.nan
        
        covar = self.covariance(method,False)
        var_market = self.variance("Market", False)

        beta = covar/var_market
        return self._clean_val(beta, is_clean)
    
    def Alpha(self, method, risk_free = 0.001, is_clean=True):
        if method == "Market":
            print("*ERROR: method는 Market이 될 수 없습니다.")
            return np.nan        
        
        mean_pfo = self.arithmetic_mean(method, False)
        beta = self.Beta(method, False)

        mean_market = self.arithmetic_mean("Market", False)
        
        alpha = mean_pfo - risk_free - beta*(mean_market -risk_free)

        return self._clean_val(alpha, is_clean)
    
    def VaR(self, method, interval = 90, is_clean=True):
        
        col_name= "{}_return".format(method)
        sort_df = self.df[[col_name]].sort_values(col_name, ascending = True)

        VaR_90 = sort_df[col_name].quantile(0.1)
        VaR_95 = sort_df[col_name].quantile(0.05)
        VaR_99 = sort_df[col_name].quantile(0.01)
        
        if interval == 90:
            return self._clean_val(VaR_90, is_clean)
        elif interval == 95:
            return self._clean_val(VaR_95, is_clean)
        elif interval == 99:
            return self._clean_val(VaR_99, is_clean)
        else:
            return [self._clean_val(VaR_90, is_clean), 
                    self._clean_val(VaR_95, is_clean), 
                    self._clean_val(VaR_99, is_clean)]        
        
    def CVaR(self, method, interval = 90, is_clean=True):
        
        col_name= "{}_return".format(method)
        sort_df = self.df[[col_name]].sort_values(col_name, ascending = True)

        if interval in [90,95,99]:
            VaR_num = self.VaR(method, interval, False)
            CVaR_int = sort_df[sort_df[col_name] <= VaR_num].mean().item()
            return self._clean_val(CVaR_int, is_clean)
        else:
            VaR_lst = self.VaR(method,'all', False)
            tmp_cvar_lst = []
            for v in VaR_lst:
                tmp_cvar = sort_df[sort_df[col_name] <= v].mean().item()
                tmp_cvar_lst.append(self._clean_val(tmp_cvar, is_clean))
            return tmp_cvar_lst
        
    def MDD(self, method, is_clean=True):
        col_name = '{}_balance'.format(method)
        m_bal_df = self.df[[col_name]].copy()
        m_bal_df["Highest_past"] = m_bal_df.cummax()
        m_bal_df["DD"] = m_bal_df[col_name]/m_bal_df["Highest_past"] - 1.0
        mdd = m_bal_df["DD"].min()

        return self._clean_val(mdd, is_clean)

    def compare_pfo_with_bm(self, method_lst, asset_name_lst, risk_free=0.001, is_clean=True):
        
        if not self._error_check(method_lst, asset_name_lst):
            print("다시 입력하세요.")
            return -1
        
        # column 생성
        col_lst = []
        for asset_name in asset_name_lst:
            for method in method_lst:
                col_lst.append("{}_{}_pfo".format(asset_name, method))
        col_lst.append('Benchmark')
        
        compare_df = pd.DataFrame([], columns = col_lst)
        
        # 계산
        ari_mean_lst = []
        ari_mean_annual_lst = []
        vol_lst = []
        vol_annual_lst = []
        sr_lst = []
        var_lst = []
        cvar_lst = []
        mdd_lst = []
        market_flag = False
        
        # asset
        for asset_name in asset_name_lst:
            for method in method_lst:
                col_name = '{}_{}'.format(asset_name, method)
                
                ari_mean_lst.append(self.arithmetic_mean(col_name, is_clean))
                ari_mean_annual_lst.append(self.annualized_art(col_name, is_clean))
                vol_lst.append(self.volatility(col_name, is_clean))
                vol_annual_lst.append(self.annual_vol(col_name, is_clean))
                sr_lst.append(self.Sharpe_ratio_annual(col_name, risk_free, is_clean))
                var_lst.append(self.VaR(col_name,'all', is_clean))
                cvar_lst.append(self.CVaR(col_name,'all', is_clean))
                mdd_lst.append(self.MDD(col_name,is_clean))
          
        # market
        col_name = 'Market'
        ari_mean_lst.append(self.arithmetic_mean(col_name, is_clean))
        ari_mean_annual_lst.append(self.annualized_art(col_name, is_clean))
        vol_lst.append(self.volatility(col_name, is_clean))
        vol_annual_lst.append(self.annual_vol(col_name, is_clean))
        sr_lst.append(self.Sharpe_ratio_annual(col_name, risk_free, is_clean))
        var_lst.append(self.VaR(col_name,'all', is_clean))
        cvar_lst.append(self.CVaR(col_name,'all', is_clean))
        mdd_lst.append(self.MDD(col_name,is_clean))       
        
    
        compare_df.loc['Arithmetic mean(monthly)'] = ari_mean_lst
        compare_df.loc['Arithmetic mean(annualized)'] = ari_mean_annual_lst

        compare_df.loc["Volatility(monthly)"] = vol_lst
        compare_df.loc["Volatility(annualized)"] = vol_annual_lst

        compare_df.loc["Sharpe Ratio"] = sr_lst
        compare_df.loc["MDD"] = mdd_lst
        
        # Alpha, Beta
        a_lst = []
        b_lst = []
        for asset_name in asset_name_lst:
            for method in method_lst:
                col_name = '{}_{}'.format(asset_name,method)
                a_lst.append(self.Alpha(col_name, risk_free, is_clean))
                b_lst.append(self.Beta(col_name,is_clean))
            
        a_lst.append(np.nan)
        b_lst.append(np.nan)
        
        compare_df.loc["Alpha"] = a_lst
        compare_df.loc["Beta"] = b_lst
        
        # VaR, CVaR
        interval_lst = [90,95,99]
        for i in range(len(interval_lst)):
            tmp_var = []
            tmp_cvar = []
            for k in var_lst:
                tmp_var.append(k[i])
            for ck in cvar_lst:
                tmp_cvar.append(ck[i])
            
            compare_df.loc["VaR {}%".format(interval_lst[i])] = tmp_var
            compare_df.loc["CVaR {}%".format(interval_lst[i])] = tmp_cvar
        self.compare_df = compare_df
        return compare_df


