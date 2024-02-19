# Encode a few constant smile_strings

PC61BM = 'COC(=O)CCCC1(C23C14C5=C6C7=C8C5=C9C1=C5C%10=C%11C%12=C%13C%10=C%10C1=C8C1=C%10C8=C%10C%14=C%15C%16=C%17C(=C%12C%12=C%17C%17=C%18C%16=C%16C%15=C%15C%10=C1C7=C%15C1=C%16C(=C%18C7=C2C2=C%10C(=C5C9=C42)C%11=C%12C%10=C%177)C3=C16)C%14=C%138)C1=CC=CC=C1'
PC71BM = 'COC(=O)CCCC1(C23C14C5=C6C7=C8C9=C1C%10=C%11C9=C9C%12=C%13C%14=C%15C%16=C%17C%18=C%19C%20=C%21C%22=C%23C%24=C%25C%26=C%27C%28=C(C%14=C%14C%12=C%11C%11=C%14C%28=C%26C%12=C%11C%10=C%10C%12=C%25C%23=C%11C%10=C1C7=C%11C%22=C6C4=C%21C%19=C2C%17=C1C%15=C%13C2=C9C8=C5C2=C31)C%16=C%18C%27=C%24%20)C1=CC=CC=C1'
ICBA = 'C1C2C3=CC=CC=C3C1C45C26C7=C8C9=C1C2=C3C8=C8C6=C6C4=C4C%10=C%11C%12=C%13C%14=C%15C%16=C%17C(=C1C1=C2C2=C%18C%19=C%20C2=C3C8=C2C%20=C(C%10=C26)C2=C%11C3=C%13C%15=C6C%17=C1C%181C6(C3=C2%19)C2CC1C1=CC=CC=C21)C1=C%16C2=C%14C(=C%124)C5=C2C7=C19'
C60 = 'C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23'

FULLERENE_LIST = ['PCBM', 'PC_{70}BM', 'PCBM[70]', 'PC_{71}BM', 'PC_{61}BM', 'PC_{60}BM', 'PC70BM', 'PC71BM', 'PC61BM', '[6,6]-phenyl-C61-butyric acid methyl ester', \
             '[6,6]-phenyl C_{61} butyric acid methyl ester', '[6,6]-phenyl-C_{61}-butyricacid methyl ester', 'PC_{61}BH', '[6,6]-phenyl-C71-butyric acid methyl ester', \
             '[6,6]-phenyl C_{71} butyric acid methyl ester', '[6,6]-phenyl-C_{71}-butyricacid methyl ester', '[6,6]-phenyl C71-butyric acid methyl ester', \
             'PC60BM', '[6,6]-phenyl-C71-butyric acid methylester', '[6,6]-phenyl C61-butyric acid methyl ester', '[6,6]-phenyl-C_{61}-buytyric acid methyl ester', '[6,6]-phenyl-C61-butyric acid methyl', '[6,6]-phenyl-C_{61}-butyric acid methyl ester', \
             '[6,6]-phenyl-C_{71}-butyric acid methyl-ester', '(6,6)-phenyl-C71-butyric acid methyl ester', '6,6-phenyl-C_{71}-butyric acid', '[6,6]-phenyl-C_{71}-butyric acid methyl ester', \
             'ICBA', 'indene-C_{60}', 'C_{60}', 'IC_{70}BA', 'indene-C70']