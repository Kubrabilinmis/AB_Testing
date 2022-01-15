"""
                                                                          AB Test Project

Facebook kısa süre önce mevcut maximum bidding adı verilen teklif verme türüne alternatif olarak yeni bir teklif türü olan average bidding’i tanıttı.
Müşterilerimizden biri   olan  bombabomba.com, bu yeni özelliği test etmeye karar verdi ve averagebidding’in maximumbidding’den daha fazla dönüşüm getirip getirmediğini anlamak
için bir A/B testi yapmak istiyor.

Veri Seti Hikayesi :  bombabomba.com’un web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan 
gelen kazanç bilgileri yer almaktadır.  Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır.

Değişkenler :
- Impression – Reklam görüntüleme sayısı
- Click – Tıklama
Görüntülenen reklama tıklanma sayısını belirtir.
- Purchase –Satın alım
Tıklanan reklamlar sonrası satın alınan ürün sayısını belirtir.
- Earning –Kazanç
Satın alınan ürünler sonrası elde edilen kazanç

"""


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

control.head()
test.head()

test.describe()          # n = 40
control.describe()       # n = 40

control["Purchase"].mean()   # max bidding        # 550.89
test["Purchase"].mean()      # average bidding    # 582.10


# Görev 1: A/B testinin hipotezini tanımlayınız.
# 1 - Hipotezi kurma
# H0: M1 = M2    Maxsimum bidding ile Average Bidding arasında   istatistiksel olarak anlamlı farklılık yoktur
# H1: M1!= M2    ....  fark vardır


# 2 - Varsayım Kontrolü
# 1.Normallık varsayımı:
# Ho : M1=M2   normallik varsayımı sağlanmaktadır.
# H1= M1!=M2   ... sağlanmamaktadır.

test_stat, pvalue = shapiro(control["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p value 0.5891 > 0.05 olduğundan H0 reddedilemez. Control grubu(Maxsimum Bidding)  için  normallik varsayımı sağlanmaktadır


test_stat, pvalue = shapiro(test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# sonuc p value 0.1541 > 0.05 olduĞundan H0 reddedilemez. Test grubu(Average Bidding) için normallik varsayımı sağlanmaktadır


# 2. Varyans Homojenlıgı varsayımı
# H0: Varyanslar Homojendir.
# H1: Varyanslar Homojen Değildir.

test_stat, pvalue = levene(control["Purchase"],
                           test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# pvalue 0.1083 old. H0 reddedilemez. Varyanslar Homojendır.



# Bağımsız İki Örneklem T Testi (Parametrik Test)
test_stat, pvalue = ttest_ind(control["Purchase"],
                              test["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# 0.3493 p-value > 0.05 dir. H0 reddedilemez. Gruplar arasında  olarak istatistiksel olarak anlamlı farklılık yoktur.


# CTR (Tıklanma Oranı )
control["Click"].sum()/ control["Impression"].sum()
test["Click"].sum() / test["Impression"].sum()


# H0: Maxsimum bidding'in (Tıklanma/Görüntülenme) oranı  Average bidding(Tıklanma/Görüntülenme) oranları arasında anlamlı bir farklılık yoktur.
# H1:  .....  vardır.

clicks = np.array([control["Click"].sum(),test["Click"].sum()])
impressions = np.array([control["Impression"].sum(),test["Impression"].sum()])

proportions_ztest(count=clicks, nobs= impressions)
# p-value < 0.05 olduğu için H0 reddedilir. % 95 güven ile maxsimum bidding'in (tıklanma/görüntülenme) oranı average bidding'in (tıklanma/görüntülenme) oranları arasında 
#anlamlı bir farklılık vardır. Maxsimum Bidding oranı Average Bidding oranından büyüktür.



# GÖREV 2 : Çıkan test sonuçlarının istatistiksel olarak anlamlı olup olmadığını yorumlayınız.
# Parametrık t test sonucunda p-value değeri 0.05 ten küçük olmadığından H0 reddedilemez. Yani maxsimum bidding ve average bidding arasında istatistiksel olarak anlamlı bır 
# farklılık yoktur.
# Yapılan oran testi sonucunda maxsimum bidding ve average bidding oranları arasında anlamlı farklılık ortaya çıkmıştır.



# GÖREV 3: Hangi testleri kullandınız? Sebeplerini belirtiniz.
# Varsayım Kontrollerınde normallık varsayımı saglandığı için çünkü cıkan sonucu p-value > 0.05 olduğundan reddedemedik .
# Varyans varsayımına bakıldığında varyans homojenliğinde sonuç yine p-value > 0.05 olduğundan dolayı reddemedik . Sonuç olarak iki varsayımımızda sağlandıgı için bağımsız iki
# örneklem t testi uyguladık.



# GÖREV 4: Görev 2’de verdiğiniz cevaba göre, müşteriye tavsiyeniz nedir?
# Control(Maksimum teklif) ve Test(Ortalama teklif) satın alma ortalamalarına baktığımızda matematiksel olarak bir farklılık vardır . Fakat bu farklılığın şans eserimi ortaya 
# çıkıp çıkmadığı bilinmemektedir. Parametrik bir test olan bağımsız iki örneklem t testine göre eski ve yeni sistemlerin getirileri arasında % 95 güvenle bir fark çıkmamıştır. 
# Bu durumda biz getirisi ortalaması yüksek olan sistemide seçebiliriz. Yada iki grubun tıklanma oranı arasında anlamlı bir farklılığını test edebiliriz. Maxsimum bidding ve 
# Average bidding tıklanma oranlarına baktığımızda maxsimum bidding in average bidding e göre daha yüksek olduğu görülmektedir. Bu farklılığın şans eseri ortaya çıkıp çıkmadığını 
# araştırmak için oran testi yapıldı. Oran testi uyguladığımızda % 95 güven ile teklif verme türleri oranları arasında anlamlı bir farklılık çıkmıştır.İki grubun oranlarına 
# baktığımızda sonuç maxsimum biddingin daha yüksek olduğunu göstermektedir.

