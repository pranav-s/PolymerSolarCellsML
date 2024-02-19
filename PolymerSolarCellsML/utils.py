def date_conversion(doc):
    year, month, day = doc['year'], doc['month'], doc['day']
    month_dict = {1: 31, 2:28, 3: 31, 4:30, 5:31, 6: 30, 7:31, 8:31, 9: 30, 10:31, 11:30, 12: 31}
    day_count = sum(month_dict[key] for key in range(1, month))
    assert day_count<=365
    year_fraction = (day_count+day)/365
    # print(year+year_fraction)
    return year+year_fraction

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def config_plots(mpl):
    mpl.rc('figure', titlesize=20, figsize=(7, 6.5))
    mpl.rc('font', family='Palatino Linotype', size=22, weight='bold')
    # mpl.rc('font', size=24)
    mpl.rc('xtick', labelsize=18, direction='in')
    #mpl.rcParams['xtick.major.size'] = 20
    #mpl.rcParams['xtick.major.width'] = 4
    mpl.rc('xtick.major', size=8, width=2)
    mpl.rc('ytick.major', size=8, width=2)
    mpl.rc('ytick', labelsize=18, direction='in')
    mpl.rc('axes', labelsize=20, linewidth=2.5, labelweight="bold")
    mpl.rc('savefig', bbox='tight', dpi=300)
    mpl.rc('lines', linewidth=1, markersize=5)
    mpl.rc('legend', fontsize=14)