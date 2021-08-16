import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import r2_score as r2
sb.set()
sb.set_style("darkgrid")
np.random.seed(2)

pageSpeeds=np.random.normal(3,1,100)
PurchaseAmount=np.random.normal(50,30,100)/pageSpeeds

train80=pageSpeeds[:80]
test20=pageSpeeds[80:]

train80Y=PurchaseAmount[:80]
test20Y=PurchaseAmount[80:]

fig,axs=plt.subplots(2,3)
TRain1=sb.scatterplot(x=pageSpeeds,y=PurchaseAmount,ax=axs[0,0])
TRain2=sb.scatterplot(x=train80,y=train80Y,ax=axs[0,1])
TRain3=sb.scatterplot(x=test20,y=test20Y,ax=axs[0,2])

X=np.array(train80)
Y=np.array(train80Y)

X2=np.array(test20)
Y2=np.array(test20Y)



poly=np.poly1d(np.polyfit(X,Y,6))
poly2=np.poly1d(np.polyfit(X2,Y2,3))


sb.lineplot(x=np.linspace(0,6,80),y=poly(np.linspace(0,6,80)),ax=axs[0,1])
sb.lineplot(x=np.linspace(0,5,20),y=poly(np.linspace(0,5,20)),ax=axs[0,2])
sb.lineplot(x=np.linspace(0,6,80),y=poly(np.linspace(0,6,80)),ax=axs[1,1])
sb.lineplot(x=np.linspace(0,5,20),y=poly(np.linspace(0,5,20)),ax=axs[1,2])

rsq=r2(Y2,poly(X2))
rsq2=r2(Y,poly(X))
print(rsq,rsq2)
plt.show()