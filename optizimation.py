from functools import partial 
import numpy as np
import math
def con_non_wrap(x):#不能少，is_feasible要接受一个函数参数
    return np.array([0])
def obj_wrapper(func,args,kwargs,x):
    return func(x,*args,**kwargs)
def is_feasible_wrapper(func,x):
    return func(x)>=0
def f_cons_wrapper(f_cons,args,kwargs,x):
    return np.array(f_cons(x,*args,**kwargs))
#选取更大的种群数量能够是值更易收敛到全局最优，亲测有效，效果比增加迭代次数要好很多
def mypso(func,lb,ub,f_cons=None,args=(),kwargs={},swarmsize=500,maxiter=100,minstep=1e-8,minfunc=1e-8):
    """
    func:优化函数
    lb:下界
    ub:上界
    args,kwargs:函数中需要固定的值
    f_cons:约束函数
    swarmsize:种群数量
    w:学习速率
    phip/g:粒子的最佳位置,群的最佳位置
    maxiter:最大迭代次数
    minstep:最小步长
    minfunc:优化目标值
    返回
    g:群的最佳位置
    gf:对应的值 
    """
    assert len(ub)==len(lb), '格式错误，上下界维度应该相等'
    assert hasattr(func, '__call__'), '无效的函数句柄'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(lb<=ub),'输入不规范，下界不能超过上界'
    vhigh=np.abs(lb-ub)
    vlow=-vhigh
    obj=partial(obj_wrapper,func,args,kwargs)
    if f_cons is None:
        cons=con_non_wrap
    else:       
        cons=partial(f_cons_wrapper,f_cons,args,kwargs)
    is_feasible=partial(is_feasible_wrapper,cons)
    #初始化种群
    s=swarmsize
    d=len(ub)
    x=np.random.rand(s,d)#（0，1）随机分布模拟粒子的初始位置
    v=np.zeros_like(x)#初始速度
    p=np.zeros_like(x)#最佳粒子位置
    fx=np.zeros(s)#粒子的函数值
    ff=np.zeros(s,dtype=bool)
    fp=np.ones(s)*np.inf#最佳位置的函数值
    g=[]#全局最佳
    fg=np.inf#全局最佳值初始化
    x=lb+x*(ub-lb)
    for i in range(s):
        fx[i]=obj(x[i,:])
        ff[i]=is_feasible(x[i,:])
    #更新位置，寻找 比上一次迭代全局最优更好的位置，并且满足约束函数
    p_update=np.logical_and((fx<fp),ff)
    p[p_update,:]=x[p_update,:].copy()
    fp[p_update]=fx[p_update]
    #在fp中寻找最小值
    i_min=np.argmin(fp)
    if fp[i_min]<fg:
        fg=fp[i_min]
        g=p[i_min].copy()#g 随着 p改变而改变
    else:
        g=x[0,:].copy()#因为是初始值所以可以任选一个位置
    #初始速度，并确保不会越界
    v=vlow+np.random.rand(s,d)*(vhigh-vlow)
    it=1
    #开始迭代
    while it<maxiter:
    #随机速度权值
        rp=np.random.uniform(s,d)
        rg=np.random.uniform(s,d)
        #更新速度
        v=0.5*v+0.5*rp*(p-x)+0.5*rg*(g-s)
        x=x+v
        #校正越界粒子
        xx=x>ub
        yy=x<lb
        x=x*(~np.logical_or(xx,yy))+ub*xx+lb*yy
        for i in range(s):
            fx[i]=obj(x[i,:])
            ff[i]=is_feasible(x[i,:])
        #再次迭代
        p_update=np.logical_and((fx<fp),ff)
        fp[p_update]=fx[p_update]
        p[p_update,:]=x[p_update,:]
        i_min=np.argmin(fp)
        if fp[i_min]<fg:
            print('新的最优位置{}{}{}'.format(it,p[i_min,:],fp[i_min]))
            p_min=p[i_min,:]
            step=np.sqrt(np.sum(p-p_min)**2)
            #最优值在点的小邻域内波动，说明值已经收敛
            if np.abs(fg-fp(i_min))<minfunc:
                return g,fg
            elif step<minstep:
                return g,fg
            else:
                g=p_min
                fg=fp[p_min]
        it+=1
    return g,fg
def simann(func,lb,ub,f_cons=None,args=[],kwargs={},T_start=1000,T_end=1,delta=0.95,maxiter=500,minstep=1e-8,minfunc=1e-8):
    assert len(ub)==len(lb), '格式错误，上下界维度应该相等'
    assert hasattr(func, '__call__'), '无效的函数句柄'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(lb<=ub),'输入不规范，下界不能超过上界'
    if f_cons is None:
        cons=partial(con_non_wrap,func,args,kwargs)
    else:       
        cons=partial(f_cons_wrapper,f_cons,args,kwargs)
    is_feasible=partial(is_feasible_wrapper,cons)
    obj=partial(obj_wrapper,func,args,kwargs)
    d=len(ub)
    fg=np.inf
    p_up=np.random.rand(d)
    while T_end<T_start:
        x=np.random.rand(d)
        x=lb+x*(ub-lb)
        fx=obj(x)
        ff=is_feasible(x)
        if ff==True:
            for i in range(maxiter):
             new_x=x+np.random.rand()
             n_fx=obj(new_x)
             n_ff=is_feasible(new_x)
             if n_fx<fx and n_ff==True:
                fx=n_fx
                if n_fx<=fg:
                    fg=n_fx
                    p_up=new_x
             elif n_fx>fx and n_ff==True:
                 p=math.exp(-(n_fx-fx)/T_start)
                 if p>np.random.rand():
                    fg=min(fg,fx)
                    p1=x
                    fx=n_fx
            #记录迭代的最小值
             
             fg=min(fg,fx)
        T_start=T_start*delta
    return p_up,fg
                
                
    
    
    