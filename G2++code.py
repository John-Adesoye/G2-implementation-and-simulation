#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:40:17 2019

@author: john
"""

import numpy as np
import pandas as pd
import requests
import bs4 as bs
import datetime
import dateutil
import scipy.stats as si
from dateutil.parser import parse
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ZeroRate():
    '''ZeroRate object, this excludes forwards from last code'''
    def __init__ (self, loc='http://www.wsj.com/mdc/public/page/2_3020-treasury.html'):
        self.loc = loc
    def bootStrp(self):
        url = self.loc
        html = requests.get(url)
        soup = bs.BeautifulSoup(html.text, 'html.parser')
        l=soup.find("div",{"class":"tbltime"})
        for span in l.findAll('span'): #Keeps only the last obs in this class, which is the date
            date=span.text
        dte=parse(date)
        today=pd.to_datetime(dte.strftime('%Y-%m-%d'))
        data=pd.read_html(self.loc) 
        bills=data[2] 
        bonds=data[1] #Selects the Treasury Notes/Bonds table.
        bills.columns=bills.iloc[0] #Renames the bills header to be the names in row 0.
        bonds.columns=bonds.iloc[0] #Renames the bonds header to be the names in row 0.
        bills=bills.drop([0])
        bonds=bonds.drop([0])
        dt=parse(date) #Converts the date format.
        today=pd.to_datetime(dt.strftime('%Y-%m-%d')) 
        yrlen=365 
        bonds.Maturity=pd.to_datetime(bonds.Maturity)
        bills.Maturity=pd.to_datetime(bills.Maturity)
        bonds.Askedyield=pd.to_numeric(bonds.Askedyield)
        bills.Askedyield=pd.to_numeric(bills.Askedyield)
        bonds=bonds[(bonds.Maturity-datetime.timedelta(yrlen))>today]
        bonds=bonds[bonds.Maturity != bonds.Maturity.shift(1)]
        bills=bills[bills.Maturity != bills.Maturity.shift(1)]
        bonds.index=np.arange(1,len(bonds)+1)
        bills['Ttm']=pd.to_numeric((bills.Maturity-today)/datetime.timedelta(yrlen))
        bonds['Ttm']=pd.to_numeric((bonds.Maturity-today)/datetime.timedelta(yrlen))
        bills['ZeroPrice']=1./(1.+(bills.Askedyield/100)*bills.Ttm) #Treasury bill prices.
    
        bonds['ZeroPrice']=pd.to_numeric(bonds.Asked)/100 #sets the quoted price
        bonds.Coupon=pd.to_numeric(bonds.Coupon)
        i=1 #Sets the bond index counting variable to the first obs.
        while i<=len(bonds):
            s=np.floor(pd.to_numeric((bonds.Maturity[i]-today)/datetime.timedelta(yrlen))*2)
            while ((bonds.Maturity[i]-dateutil.relativedelta.relativedelta(months=s*6)>today) & (bonds.Maturity[i]-dateutil.relativedelta.relativedelta(months=s*6)<bonds.Maturity[i])):
                cpndate=bonds.Maturity[i]-dateutil.relativedelta.relativedelta(months=s*6)
                if pd.to_numeric((cpndate-today)/datetime.timedelta(yrlen))<1:
                    absdif=abs(bills.Maturity-cpndate)
                    df=bills.ZeroPrice[absdif.idxmin()]
                else:
                    absdif=abs(bonds.Maturity-cpndate)
                    df=bonds.ZeroPrice[absdif.idxmin()]
                if s==np.floor(pd.to_numeric((bonds.Maturity[i]-today)/datetime.timedelta(yrlen))*2):
                    bonds.ZeroPrice[i]=bonds.ZeroPrice[i]+((bonds.Coupon[i]/100)/2)*(1-pd.to_numeric((cpndate-today)/datetime.timedelta(30*6)))
                bonds.ZeroPrice[i]=bonds.ZeroPrice[i]-((bonds.Coupon[i]/100)/2)*df
                s=s-1
            bonds.ZeroPrice[i]=bonds.ZeroPrice[i]/(1+((bonds.Coupon[i]/100)/2))
            i=i+1
        bonds['ZeroYield']=(1/(bonds.ZeroPrice**(1/bonds.Ttm))-1)*100
        zero=pd.DataFrame((bills.Askedyield).append(bonds.ZeroYield))
        zero.columns=['Yield']
        zero['Price']=(bills.ZeroPrice).append(bonds.ZeroPrice)
        zero['Maturity']=(bills.Maturity).append(bonds.Maturity)
        zero.index=np.arange(1,len(zero)+1)
        zero['MA']=zero.Yield.rolling(window=12,center=True,min_periods=0).mean()
        return bonds[['Ttm','ZeroPrice']].append(bills[['Ttm','ZeroPrice']])

trea_dat = ZeroRate()
z = trea_dat.bootStrp()

def getData(url):
    '''g=Gets most recent volcube data from the web'''
    if datetime.datetime.today().weekday() == 5:
        td = datetime.timedelta(days=1)
    elif datetime.datetime.today().weekday() == 6:
        td = datetime.timedelta(days=2)
    else:
        try:
            td = datetime.timedelta(days=0)
            today = str((datetime.datetime.today()-td).strftime('%Y%m%d'))
            data = pd.read_csv(url+today+'.csv',sep=',')
            return data
        except:
            td = datetime.timedelta(days=1)
            today = str((datetime.datetime.today()-td).strftime('%Y%m%d'))
        else:
            td = datetime.timedelta(days=3)
    today = str((datetime.datetime.today()-td).strftime('%Y%m%d'))
    url = url[:-12]
    data = pd.read_csv(url+today+'.csv',sep=',')
    return data

def BS(s, exp, logV):
    '''calculated black scholes price'''
    price = s * (si.norm.cdf(1/2*logV*np.sqrt(exp),0.0,1.0) - si.norm.cdf(-1/2*logV*np.sqrt(exp),0.0,1.0))
    return price

def clean(url='ftp://ftp.cmegroup.com/irs/CME_ATM_VolCube_20190415.csv'):
    '''modifies data from source for model use'''
    data = getData(url)
    data['aslice'] = data.Expiry.str.slice(0,1)
    data['bslice'] = data.Expiry.str.slice(1,2)
    data.UnderlyingTenor = data.UnderlyingTenor.apply(lambda x: '0'+x if len(x)==2 else x)
    data = data.sort_values(['bslice','aslice','UnderlyingTenor'])
    data['ttm'] = data.bslice.apply(lambda x: 1/12 if x=='M' else 1)*pd.to_numeric(data.aslice)
    data['Tenor'] = data.UnderlyingTenor.str.slice(0,2).astype('int64')
    data['BS'] = BS(data.Strike, data.ttm, data.LogNormalVol)
    data.index = np.arange(len(data))
    return data
    



def genttm():
    '''looks ups the ZeroPrice from bootstrapped data'''
    data_ttm  = {'1M': np.arange(1/12,362/12,6/12), '3M':np.arange(3/12,364/12,6/12),\
          '6M':np.arange(6/12,367/12,6/12), '12M':np.arange(12/12,373/12,6/12),\
          '24M':np.arange(24/12,385/12,6/12)}
    return data_ttm

params = np.array([0.02,0.02,0.02,0.495,0.001,0.001,0.015])
global gttm
gttm = genttm()

def optimize(params,tau=0.5):
    '''optmization function, G2++ formular'''
    s, a, b, rho, eta, x, y = params
    swpdt = clean()
    mktp = np.array(swpdt.OptionPrice).reshape(35,1)
    G2pp = []
    exp = np.array(swpdt.Expiry).reshape(35,1)
    tenor = np.array(swpdt.Tenor).reshape(35,1)
    for i in range(len(exp)):
        if exp[i] == '1M':
            ttm = gttm['1M']
        elif exp[i] == '3M':
            ttm = gttm['3M']
        elif exp[i] == '6M':
            ttm = gttm['6M']
        elif exp[i] == '1Y':
            ttm = gttm['12M']
        elif exp[i] == '2Y':
            ttm = gttm['24M']
        ttm = np.array(ttm).reshape(len(ttm),1)
        ttm = ttm[1:(int((tenor[i]))*2)+1]
        VtT =s**2/a**2*(ttm+(2/a)*np.exp(-a*ttm)-1/(2*a)*np.exp(-2*a*ttm)-3/(2*a))+eta**2/\
        b**2*(ttm+(2/b)*np.exp(-b*ttm)-1/(2*b)\
        *np.exp(-2*b*ttm)-3/(2*b))+2*rho*s*eta/(a*b)*(ttm+(np.exp(-a*ttm)-1)/a+(np.exp(-b*ttm)-1)/b-(np.exp(-(a+b)*ttm)-1)/(a+b))
#        AtT = 1/2 * (VtT) - ((1-(np.exp(-a*ttm))/a)*x) - (((1-(np.exp(-b*ttm))/b)*y))
        PtT = np.exp(-(1-np.exp(-a*ttm))/a*x-(1-np.exp(-b*ttm))/b*y+1/2*VtT)
        PtT = PtT.sum()
        ps = swpdt.BS[i] * tau * PtT
        G2pp.append(ps)
    G2pp = np.array(G2pp).reshape(35,1)
    mini = (G2pp - mktp)**2
    return mini.sum()

def LP(params):
    '''optmization code based on scipy library'''
    bnds = [(0,None),(1e-9,None),(1e-9,None),(-1.,1.),\
            (0,None),(None,None),(None,None)]
    sol = minimize(optimize,params,bounds=bnds,method = 'TNC')
    return sol.x

def G2ppsim(params=params, rate=0.0243):
    '''simulation of G2++ interest rate process'''
    sigma, a, b, rho, eta, x, y = LP(params)
    dt = 1/252
    ttm = np.matrix(np.arange(dt,253/252,dt)).T
    simlen = len(ttm)
    nsim = 100
    rsimmat=np.matrix(np.zeros((simlen,nsim+1)))
    rsimmat[:,0]=ttm[0:simlen]
    rsimmat[0,1:nsim]=np.matrix(np.ones(nsim-1))*rate
    for i in range(251):
        dw1 = np.matrix(np.random.normal(0,1,100)) * np.sqrt(dt)
        dw2 = np.matrix(np.random.normal(0,1,100)) * np.sqrt(dt)
        dxt = -a*rsimmat[i,1:nsim+1]*dt  + sigma*dw1
        dyt = -b*rsimmat[i,1:nsim+1]*dt + eta*rho*dw1 + eta*np.sqrt(1-rho**2)*dw2
        dr = dxt + dyt
        rsimmat[i+1,1:nsim+1] = rsimmat[i,1:nsim+1] + dr
    i=1
    while i<nsim:
        plt.plot(rsimmat[:,0],rsimmat[:,i])
        i=i+1
    plt.title("100 1-year interest rate paths based on G2++ model"+"\n("+str(datetime.datetime.today().date())+")")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Interest rate (G2++ model)")
    plt.show()

#Results from above codes
G2ppsim()
sigma, a, b, rho, eta, x, y = LP(params)
names = ['sigma', 'a', 'b', 'rho', 'eta', 'x', 'y']
for i in names:
    print(i+':',vars()[i])




        
    





