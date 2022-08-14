import pandas as pd
import numpy as np


def add_market_df(backtest_df_lst, market, amount):
    
    """
    === Description ===
    여러 자산들의 백테스트 결과와 해당 기간동안 벤치마크 수익률 데이터프레임을 합쳐 하나의 데이터프레임으로 생성
    output은 성능평가(Performance Evaluation)와 시각화(Visualization)에 사용됨

    * parameter *
    backtest_df_lst: bactest후 나온 dataframe들을 담은 리스트 / type: list / ex) [df1, df2]
    market: benchmark dataframe / type:DataFrame
    amount: 투자금액 (backtest_df_lst 내 모든 초기투자금액과 동일해야 하며, 그렇지 않으면 에러 발생)
    """

    # ERROR 체크
    if not isinstance(backtest_df_lst, list):
        print("[ERROR]\n list형식으로 입력하세요.")
        return -1
    
    if not backtest_df_lst:
        print("[ERROR]\n backtest_df_lst에 데이터를 입력해주세요")
        return 0
    elif len(backtest_df_lst) > 1:
        # 1차 사이즈 확인
        size = len(backtest_df_lst[0])
        for i in range(1, len(backtest_df_lst)):
            if len(backtest_df_lst[i]) != size:
                print("[ERROR]\n 데이터의 총 투자기간이 다릅니다.")
                return 1
        
            if size != sum(backtest_df_lst[0].index == backtest_df_lst[1].index):
                print("[ERROR]\n 데이터의 세부 투자기간이 다릅니다.")
                return 2
    
    for i in range(len(backtest_df_lst)):
        col_name = ' '
        for col in backtest_df_lst[i]:
            if 'balance' in col:
                col_name = col
                break
        bt_amount = backtest_df_lst[i].loc['Initial',col_name]
        if amount != bt_amount:
            split_lst = col_name.split('_')
            print("[ERROR]\n {}자산의 초기 balance가 {}이므로 입력된 {}와 다릅니다.".format(split_lst[0], bt_amount, amount))
            return 3

    float_df_lst= []
    for i in range(len(backtest_df_lst)):
        float_df_lst.append(backtest_df_lst[i].astype('float64'))

    res = pd.concat(float_df_lst, axis = 1)
    
    res["Market_return"] = market[backtest_df_lst[0].index[1]:backtest_df_lst[0].index[-1]]
    res["Market_balance"] = amount

    for i in res.index[1:]:
        amount *= (res.Market_return.loc[i] + 1)
        res.loc[i,"Market_balance"] = amount
        
    res.loc['Initial','Market_return'] = 0
    
    return res

