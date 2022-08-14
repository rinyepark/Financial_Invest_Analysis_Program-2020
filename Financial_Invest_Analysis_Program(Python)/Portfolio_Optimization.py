import numpy as np
from scipy import stats

import cvxopt as opt
from cvxopt import *
from cvxopt import blas, solvers

import cvxpy as cp

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd

# !pip install PyPortfolioOpt
import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier

import scipy.optimize as sco

from plotly.subplots import make_subplots
from sklearn import linear_model


class Portfolio_Optimization:

    '''
    Description: 포트폴리오 최적화 모델
     (Constructor: param)
      - est_lst: lookback 기간 동안의 estimate
          est_lst[0]: mean lst ex) [0.5,0.3, ..., 0.7] 
          est_lst[1]: var lst ex) [...]
          est_lst[2]: cov matrix ex) [[...],[...],...]
          est_lst[3]: holding-period return 
      - ret: lookback 기간 바로 이후 자산 구성
     (Constructor: default param)
      - step_size: Lasso_n에 사용되는 step_size 목록
    
    Method
     (Public)
      - Equal_weight: 동일 가중치 모델
      - Mean_variance: 포티플리오의 risk를 최소화하는 평균-분산 모델
      - Risk_parity: 각 자산마다 risk를 동등하게 갖는 Risk parity 모델
      - Ridge: Ridge규제 추가 모델
      - Lasso: Lasso규제 추가 모델
      - Lasso_n: Lasso규제를 이용해 n개의 자산만으로 포트폴리오 구성하는 최적의 람다를 이용해 가중치를 계산하는 모델
      - Factor1: Mkt Factor에 의해 영향 받는 상위 (자산군 수)/3 개의 자산군
      - Factor2: SMB Factor에 의해 영향 받는 상위 (자산군 수)/3 개의 자산군
      - Factor3: HML Factor에 의해 영향 받는 상위 (자산군 수)/3 개의 자산군
      - AllSeason: All season 포트폴리오
      - GB:  Golden butterfly 포트폴리오
      
      - regression: Factor와 자산을 이용한 회귀분석 진행
      
     (Private)
      - _L1_norm: Lasso 규제 이용함수
     
    '''
    
    def __init__(self,est_lst,ret):
        self.est_lst = est_lst
        self.ret = ret
        self.step_size = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  

    def Equal_weight(self):
        weight_dict = {}
        for i in self.est_lst[0].index:
            weight_dict.update({i:1/len(self.ret)})
        weight_lst = pd.Series(weight_dict)
        return weight_lst
    
    
    def Mean_variance(self):
      
        n=len(self.est_lst[0])
        
        def minvar(wt):
            ## local에서 이부분 동작 안해서 수정했습니다.
            res_minvar = np.array(np.sqrt(wt.T @ np.matrix(self.est_lst[2]) @ wt)).flatten()
            return res_minvar


        w = np.array([1/n]*n) 
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        bnds = ((0,1),) *n

        opt = sco.minimize(minvar, w, method='SLSQP', bounds=bnds, constraints=cons)
        weight_dict = {}
        idx=0
        for col in self.est_lst[0].index:
            weight_dict.update({col:opt['x'][idx]})
            idx+=1
        weight_lst = pd.Series(weight_dict)
        return weight_lst


    def Risk_parity(self):
        n=len(self.est_lst[0])
        #risk를 동일하게 나누고 차이의 제곱합을 return
        def RiskParity(wt):
            VarianceMatrix = self.est_lst[2].multiply(wt, axis = 1).multiply(wt, axis = 0) 
            RCratio = VarianceMatrix.sum() / VarianceMatrix.sum().sum()
            return np.sum(((1.0/n)-np.array(RCratio))**2)

        w = np.array([1/n]*n) 
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        bnds = ((0,1),) *n

        #option = {'maxiter': 100, 'ftol': 1e-60}     # ftol은 거의 0에 가까운 목표값, maxiter는 최대반복횟수
        #opt = sco.minimize(RiskParity, w, method='SLSQP', bounds=bnds, constraints=cons,options=option)

        opt = sco.minimize(RiskParity, w, method='SLSQP', bounds=bnds, constraints=cons)

        # opt.x 를 이용하여 risk가 적절히 나뉘었는지 확인
        #VarianceMatrix = est_lst[2].multiply(opt.x, axis = 1).multiply(opt.x, axis = 0)     
        #RCratio = VarianceMatrix.sum() / VarianceMatrix.sum().sum()
        #print("----risk----")
        #print(RCratio)

        weight_dict = {}
        idx=0
        for col in self.est_lst[0].index:
            weight_dict.update({col:opt['x'][idx]})
            idx+=1
        weight_lst = pd.Series(weight_dict)
        return weight_lst

        
    def Ridge(self, g=1):
        ef = EfficientFrontier(self.est_lst[0], self.est_lst[2])
        ef.add_objective(pypfopt.objective_functions.L2_reg, gamma=g)
        weight = ef.min_volatility()
        weight_lst = pd.Series(ef.clean_weights())
        return weight_lst

    def _L1_norm(self, w, k):
        return k * cp.norm(w, 1)
    
    def Lasso(self, l=1):
        ef = EfficientFrontier(self.est_lst[0], self.est_lst[2])
        ef.add_objective(self._L1_norm, k=l)
        weight = ef.min_volatility()
        weight_lst = pd.Series(ef.clean_weights())
        return weight_lst

    def Lasso_n(self, user_input, total_asset_num):
        if user_input > total_asset_num:
            print("ERROR: 사용자 입력({})이 전체 자산의 개수({})를 초과하였습니다.".format(user_input, total_asset_num))
            return np.nan
        
        LARGE = total_asset_num + 1
        
        beg = 0
        limit = 100

        fin_asset_num = 0
        fin_asset_weight = []

        mini_asset_num = 0
        mini_asset_weight = []
        mini_diff = LARGE
        
        target_lambda = 0
        inter_fin_flag = False

        same_num_limit = 5
        for i in range(len(self.step_size)):
            tmp_limit = limit

            num1 = 1
            num2 = 1
            count1 = 0
            count2 = 0
            for k in np.arange(beg, limit, self.step_size[i]):
                k = round(k,i+1)

                tmp_weight = self.Lasso(k)
                weight_num = total_asset_num-(tmp_weight==0).sum()

                diff = weight_num - user_input

                if diff > 0:
                    if mini_diff > diff and not inter_fin_flag:
                        mini_diff = diff
                        mini_asset_weight = tmp_weight
                        mini_asset_num = weight_num
                        tmp_limit = k
                        target_lambda = k
                    elif mini_diff == diff and not inter_fin_flag:
                        if num1 >= same_num_limit:
                            if target_lambda > k:
                                target_lambda = k
                                mini_diff = diff
                                mini_asset_weight = tmp_weight
                                mini_asset_num = weight_num
                                tmp_limit = k                   
                            break
                        num1 += 1
                    else:
                        count1 += 1
                        if count1 > 30:
                            break

                elif diff < 0:
                    
                    if mini_diff > abs(diff) and not inter_fin_flag:
                        mini_diff = abs(diff)
                        mini_asset_weight = tmp_weight
                        mini_asset_num = weight_num
                        target_lambda = k

                    elif mini_diff == abs(diff) and not inter_fin_flag:
                        if num2 >= same_num_limit:
                            if target_lambda > k:
                                target_lambda = k
                                mini_diff = abs(diff)
                                mini_asset_weight = tmp_weight
                                mini_asset_num = weight_num
                                tmp_limit = k
                            break
                        num2 += 1
                    else:
                        count2 += 1
                        if count2 > 30:
                            break
                        

                else:
                    tmp_limit = k
                    if not inter_fin_flag:
                        inter_fin_flag = True
                        fin_asset_num = weight_num
                        fin_asset_weight = tmp_weight
                        target_lambda = k
                    else:
                        if target_lambda > k:
                            fin_asset_num = weight_num
                            fin_asset_weight = tmp_weight
                            target_lambda = k                    
                    break
        
            if target_lambda < tmp_limit and target_lambda - self.step_size[i] >= 0:
                beg = target_lambda - self.step_size[i]
            elif target_lambda >= tmp_limit and tmp_limit - self.step_size[i] >= 0:
                beg = tmp_limit - self.step_size[i]
                

            limit = tmp_limit

        if inter_fin_flag:
            print("LASSO: 자산 {}개의 포트폴리오를 구성하기 위한 최적의 lambda는 {}입니다.\n".format(fin_asset_num,target_lambda))
            return fin_asset_weight
        else:
            print("※NOTICE: 사용자의 입력({})과 가장 유사한 {}개의 자산을 구성하였습니다.".format(user_input, mini_asset_num))
            print("LASSO:자산 {}개의 포트폴리오를 구성하기 위한 최적의 lambda는 {}입니다.\n".format(mini_asset_num, target_lambda))
            return mini_asset_weight

    def regression(self, data, ff_data) :
        X = ff_data[['Mkt-RF','SMB', 'HML']]
        Y = data[data.columns]
        # Linear Regression 부분입니다!
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)

        #print('Intercept: \n', regr.intercept_)
        #print('Coefficients: \n', regr.coef_)
        regr.coef_ = regr.coef_.T
        coef = regr.coef_.argsort(axis=1)
        #print(coef)

        top_ind = int(len(data.columns)/3)

        Mkt_RF_lst = []
        SMB_lst = []
        HML_lst = []
        for i in range(len(coef)) :
            for j in range(len(coef[i])-1,len(coef[i])-1-top_ind, -1) :
                if i == 0 :
                    Mkt_RF_lst.append(data.columns[coef[i][j]])
                elif i == 1 :
                    SMB_lst.append(data.columns[coef[i][j]])
                elif i == 2 :
                    HML_lst.append(data.columns[coef[i][j]])

        top_ind_lst = []
        top_ind_lst.append(Mkt_RF_lst)
        top_ind_lst.append(SMB_lst)
        top_ind_lst.append(HML_lst)

        return top_ind_lst

    def Factor1(self, ret, ff_data) :
        top_ind_lst = self.regression(ret, ff_data)
        
        weight_dict = {}
        for i in self.est_lst[0].index:
            if i in top_ind_lst[0]:
                weight_dict.update({i:1/len(top_ind_lst[0])})
            else :
                weight_dict.update({i:0})
        weight_lst = pd.Series(weight_dict)
        return weight_lst
    
    def Factor2(self, ret, ff_data) :
        top_ind_lst = self.regression(ret, ff_data)
        
        weight_dict = {}
        for i in self.est_lst[0].index:
            if i in top_ind_lst[1]:
                weight_dict.update({i:1/len(top_ind_lst[1])})
            else :
                weight_dict.update({i:0})
        weight_lst = pd.Series(weight_dict)
        return weight_lst
    
    def Factor3(self, ret, ff_data) :
        top_ind_lst = self.regression(ret, ff_data)
        
        weight_dict = {}
        for i in self.est_lst[0].index:
            if i in top_ind_lst[2]:
                weight_dict.update({i:1/len(top_ind_lst[2])})
            else :
                weight_dict.update({i:0})
        weight_lst = pd.Series(weight_dict)
        return weight_lst

    def AllSeason(self) :
        weight_dict = {'US BOND 7-10':0.15,'US BOND 20+':0.4,'GOLD':0.075,'COMMODITY':0.075,'S&P ETF':0.3}
        weight_lst = pd.Series(weight_dict)
        return weight_lst
    
    def GB(self) :
        weight_dict = {'US BOND 7-10':0.2,'US BOND 20+':0.2,'GOLD':0.2,'COMMODITY':0,'S&P ETF':0.4}
        weight_lst = pd.Series(weight_dict)
        return weight_lst
    
#     def _L1_L2_norm(self, w, k, gamma):
#         return k * cp.norm(w, 1) + gamma*cp.norm(w, 2)
    
#     def Elastic(self, l=1, g=1):
#         ef = EfficientFrontier(self.est_lst[0], self.est_lst[2])
#         ef.add_objective(self._L1_L2_norm, k=l, gamma= g)
#         weight = ef.min_volatility()
#         weight_lst = pd.Series(ef.clean_weights())
#         return weight_lst