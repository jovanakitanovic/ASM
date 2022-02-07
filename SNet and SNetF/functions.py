import pandas as pd
import networkx as nx
import operator
import numpy as np
import seaborn as sns
import matplotlib as plot
from collections import Counter
import powerlaw
import matplotlib.pyplot as plot
from itertools import chain, combinations
from scipy.cluster.hierarchy import dendrogram

def subreddit_with_most_participants(subreddit_participant):

    """------ pronalaženje subreddita sa najbećim brojem korisnika, (grupisanjem po subredditu a zatim uzimanjem onog sa najvećom grupom)
    -----------odnosno, taj subreddit se najvišeputa pojavljuje u obe grupe fajlova -----/BEGIN/---------------------------------------------"""

    subreddit_with_most_participants=subreddit_participant.groupby("subreddit_id").size().sort_values(ascending=False)
    print("SUBREDITI SA NAJVEĆIM BROJEM KORISNIKA")
    print(subreddit_with_most_participants)
    #print(len(subreddit_with_most_participants))

    """--------------/END/-------------------------------------------------------------------------------------------------------------"""

def author_with_most_submissions(subreddit_participant_withot_comments):
    """------ pronalaženje korisnika sa najbećim brojem objava, (grupisanjem po autorima a zatim uzimanjem onog sa najvećom grupom)
    -----------odnosno, taj autor se najvišeputa pojavljuje u fajlovima sa objavama-----/BEGIN/--------------------------------------"""

    subreddit_participant_withot_comments = pd.concat(subreddit_participant_withot_comments)
    filtered_data=subreddit_participant_withot_comments['author']!="[deleted]"
    subreddit_participant_withot_comments=subreddit_participant_withot_comments[filtered_data]

    author_with_most_submissions=subreddit_participant_withot_comments.groupby("author").size().sort_values(ascending=False)


    print("KORISNIK SA NAJVEĆIM BROJEM OBJAVA")
    print(author_with_most_submissions)
    return author_with_most_submissions

    """--------------/END/-------------------------------------------------------------------------------------------------------------"""

def average_num_author_per_subreddit(subreddit_participant,num_of_subreddits):
    print("\n")
    subreddit_user_grup_by_subreddit=subreddit_participant.groupby(["subreddit_id", "author"])

    sum=0;

    for elem in subreddit_user_grup_by_subreddit.groups:
        sum += 1
        #print(elem,aut)


    #print(subreddit_user_grup_by_subreddit.groups)

    sum=sum/num_of_subreddits
    print(f"PROSEČAN BROJ ZABELEŽENIH KORISNIKA PO SUBREDITU JE {sum}")


    print("\n")

def   subreddit_with_most_comments(subreddit_comment_author):
    """------ pronalaženje subreddita sa najbećim brojem komentara, (grupisanjem po subredditu a zatim uzimanjem onog sa najvećom grupom)
    -----------odnosno, taj subreddit se najvišeputa pojavljuje u fajlovima sa komentarima---/BEGIN/--------------------------------------"""

    subreddit_with_most_participants=subreddit_comment_author.groupby("subreddit_id").size().sort_values(ascending=False)
    print("SUBREDITI SA NAJVEĆIM BROJEM KOMENARA")
    print(subreddit_with_most_participants)
    """--------------/END/-------------------------------------------------------------------------------------------------------------"""


def author_with_most_comments(subreddit_comment_author):
    """------ pronalaženje korisnika sa najbećim brojem komentara, (grupisanjem po autorima a zatim uzimanjem onog sa najvećom grupom)
    -----------odnosno, taj autor se najvišeputa pojavljuje u obe grupe fajlova -----/BEGIN/---------------------------------------------"""

    author_with_most_comments=subreddit_comment_author.groupby("author").size().sort_values(ascending=False)
    print("KORISNIK SA NAJVEĆIM BROJEM KOMENARA")
    print(author_with_most_comments)
    return author_with_most_comments

    """--------------/END/-------------------------------------------------------------------------------------------------------------"""

def users_active_on_most_subreddits(subreddit_participant):

    """------ pronalaženje korisnika koji je aktivan na najvećem broju subredta (iskorišćen je već grupisan podatak po autoru
    ----------i subreddit-id-ju, zatim je u petlji napravljen rečnik takav da se za svakog korisnika broji koliko puta se u filtritanim
    ---------podacima pojavio. sortiranjem i uzimanjem nekoliko sa vrha se dobije ispis)-----/BEGIN/-------------------------------------"""

    dictionary_of_activityes={}

    for elem,subreddit in subreddit_participant.groups:
        if elem in dictionary_of_activityes.keys():
            dictionary_of_activityes[elem]+=1
        else:
            dictionary_of_activityes[elem]=1

    for i in range(0,15):
        elem=max(dictionary_of_activityes.items(), key=operator.itemgetter(1))
        dictionary_of_activityes[elem[0]]=0
        print(elem)

    """--------------/END/-------------------------------------------------------------------------------------------------------------"""

def piersons_correlation(author_comments,author_submissions):
    print("piersons correlation")

    author_submissions.replace()

    summary_data=pd.concat([author_submissions,author_comments],axis=1,keys=["subs","comments"])

    summary_data["subs"]=summary_data["subs"].fillna(0)
    summary_data["comments"]=summary_data["comments"].fillna(0)


    print(summary_data)
    print("\n")

    arr_comments=[]
    arr_subs=[]

    for elem in summary_data["subs"]:
        if np.isnan(elem):
            arr_subs.append(0)
        else:
            arr_subs.append(elem)

    for elem in summary_data["comments"]:
        if np.isnan(elem):
            arr_comments.append(0)
        else:
            arr_comments.append(elem)

    correlation_array_comments=np.array(arr_comments)
    correlation_array_submissions=np.array(arr_subs)

    correlation_results=np.corrcoef(summary_data["comments"],summary_data["subs"])

    print(correlation_results)

    from scipy import stats
    print(stats.pearsonr(correlation_array_submissions,correlation_array_comments))

    plot.pyplot.scatter(x=summary_data["comments"], y=summary_data["subs"])
    plot.pyplot.show()

    sns.lmplot(x="comments", y="subs",data=summary_data,fit_reg=False)
    plot.pyplot.show()

    #sns.lmplot(x="subs", y="comments",  data=summary_data,ci=None )
    #plot.pyplot.show()



def centrality_analysis(network):

    DC=nx.degree_centrality(network)
    CC=nx.closeness_centrality(network)
    BC=nx.betweenness_centrality(network)


    df1 = pd.DataFrame.from_dict(DC, orient='index', columns=['DC'])
    df2 = pd.DataFrame.from_dict(CC, orient='index', columns=['CC'])
    df3 = pd.DataFrame.from_dict(BC, orient='index', columns=['BC'])
    #df4 = pd.DataFrame.from_dict(EVC_dict, orient='index', columns=['EVC'])

    df = pd.concat([df1, df2, df3], axis=1)
    #print(df1.sort_values(["DC"],ascending=False))

    print(df.sort_values(["DC"],ascending=False).head(10))
    print(df.sort_values(["CC"],ascending=False).head(10))
    print(df.sort_values(["BC"],ascending=False).head(10))


def eigen_centrality(network):

    eigen=nx.eigenvector_centrality(network)
    df = pd.DataFrame.from_dict(eigen, orient='index', columns=['EIGEN'])

    print(df.sort_values(["EIGEN"],ascending=False))

def dominant_component(network):
    largest_cc = max(nx.connected_components(network), key=len)
    S = [network.subgraph(c).copy() for c in nx.connected_components(network)]
    network_dom = network.subgraph(largest_cc).copy()
    print(f"Dominantna komponenta ima {len(network_dom.nodes())} čvorova i {len(network_dom.edges())} grana")
    dominant_component_path = "dominantna_komponenta.net"
    #nx.write_pajek(network_dom, dominant_component_path)

def export_net(network):
    dominant_component_path = "dominantna_komponenta_SNetF.net"
    nx.write_pajek(network, dominant_component_path)

def network_assortativity(network):

    """ U prvoj petlji se prolazi kroz svaku granu i uzima se prvi čvor koji se stavlja u y data i drugi čvor koji
    se stavlja u x data tako da se na grafiku vidi kako se čvor povezuje sa svojim susedima"""

    num=0

    xdata=[]
    ydata=[]

    for i,j in network.edges:
        #num+=1
        #print(i,j,num)
        ydata.append(network.degree(i))
        xdata.append(network.degree(j))

    node_degree=[]
    neighbours_averrage_degree=[]

    for node in network.nodes:
        num+=1
        node_degree.append(network.degree(node))
        neighbours_averrage_degree.append((list(nx.average_neighbor_degree(network, nodes=[node]).values())[0]))


    print(len(neighbours_averrage_degree),len(node_degree))

    print(f"ASORTATIVNOST {nx.degree_assortativity_coefficient(network)}")

    plot.pyplot.scatter(neighbours_averrage_degree,node_degree,alpha=0.05)
    plot.pyplot.xlim(0, max(node_degree))
    plot.pyplot.ylim(0, max(neighbours_averrage_degree))
    plot.pyplot.ylabel('node degree')
    plot.pyplot.xlabel('average degree of neighbour')
    plot.pyplot.show()

    plot.pyplot.scatter(xdata, ydata, alpha=0.05)
    plot.pyplot.xlim(0, (max(node_degree)+50))
    plot.pyplot.ylim(0, (max(node_degree)+50))
    plot.pyplot.ylabel('node degree')
    plot.pyplot.xlabel('degree of neighbour')
    plot.pyplot.show()

def network_clasterization(network,randomGraphSameSize,randomGraphErdosRenyi):

    sorted_values=sorted(nx.clustering(network).items(),key=operator.itemgetter(1))
    print(sorted_values)

    print("\n")
    print(f"PROSEČAN STEPEN KLASTERISANJA {nx.average_clustering(network)}")
    print(f"GLOBALNI KOEFICIJIENT KLASTERIZACIJE {nx.transitivity(network)}")
    print("\n")

    print(f"PROSEČAN STEPEN KLASTERISANJA ERDOS-RENYI {nx.average_clustering(randomGraphErdosRenyi)}")
    print(f"GLOBALNI KOEFICIJIENT KLASTERIZACIJE ERDOS-RENYI {nx.transitivity(randomGraphErdosRenyi)}")
    print("\n")

    print(f"PROSEČAN STEPEN KLASTERISANJA RANDOM NET {nx.average_clustering(randomGraphSameSize)}")
    print(f"GLOBALNI KOEFICIJIENT KLASTERIZACIJE RANDOM NET {nx.transitivity(randomGraphSameSize)}")
    print("\n")

def katz_centrality(network,beta_reddit_com,beta):
    dictionary_katz_centrality={}

    for elem in network.nodes:
            if(elem=="t5_6"):
                dictionary_katz_centrality[elem]=beta_reddit_com
            else:
                dictionary_katz_centrality[elem]=beta


    print(network.number_of_nodes())

    lambda_max=max(nx.adjacency_spectrum(network))
    print(f"1/lambda max {1/lambda_max}")

    #katz_centrality_results = nx.katz_centrality(network, alpha=0.005, beta=1,max_iter=1000)
    katz_centrality_results = nx.katz_centrality(network, alpha=0.0005, beta=dictionary_katz_centrality,max_iter=1000)

    #print(katz_centrality_results)
    df_katzc = pd.DataFrame.from_dict(katz_centrality_results, orient='index', columns=['EVC'])
    df_katzc.sort_values(by='EVC',ascending=False,inplace=True)
    #print("\n ")
    print(df_katzc.head(10))




def power_law_distribution(network, weighted):

    plot_degree_distribution(network,weighted)
    #plot_degree_distribution(network,True)
    plot_cumulative_distribution(network)

def plot_degree_distribution(network, weighted):
    xscale = "log"
    yscale = "log"

    if weighted:
        degrees = network.degree(weight="weight")
    else:
        degrees = network.degree()

    _, deg_list = zip(*degrees)
    deg_counts = Counter(deg_list)
    print(deg_counts)
    x, y = zip(*deg_counts.items())

    plot.figure(1)

    # prep axes
    if weighted:
        plot.xlabel('weighted degree')
    else:
        plot.xlabel('degree')

    plot.xscale(xscale)
    plot.xlim(1, max(x))

    plot.ylabel('frequency')
    plot.yscale(yscale)
    plot.ylim(1, max(y))

    plot.scatter(x, y, marker='.')
    plot.show()


def plot_cumulative_distribution(network):

    """xscale = "log"
    yscale = "log"

    plot.yscale(yscale)
    plot.xscale(xscale)"""

    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    degree_count = Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())

    max_degree=max(deg)


    values, base = np.histogram(deg, bins=max_degree)
    cumulative = np.cumsum(values)
    #plot.plot(base[1:], cumulative, c='blue')
    #plot.plot(base[:-1], [float(x)/len(data) for x in len(data)-cumulative], c='blue')
    #plot.show()

    results = powerlaw.Fit(degree_sequence)

    print(results.supported_distributions)

    #results.distribution_compare("power_law","exponential")
    fig = results.plot_ccdf(color='b', linewidth=3)
    results.exponential.plot_ccdf(ax=fig, color='r', linestyle='dotted', label="expon.")
    #results.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label="lognormal")  # lognormal
    results.stretched_exponential.plot_ccdf(ax=fig, color='m', linestyle='-.', label="strec.expon.")  # stretched_exponential
    results.truncated_power_law.plot_ccdf(ax=fig, color='c', linestyle='dotted', label="trunc. power-law")  # truncated_power_law
    results.lognormal_positive.plot_ccdf(ax=fig, color='b', linestyle='dotted', label="lognor. pos.")  # lognormal_positive
    plot.legend(loc="lower left")
    plot.show()



    fig = results.plot_ccdf(color='b', linewidth=3)
    results.power_law.plot_ccdf(ax=fig, color='y', linestyle='--',label="power_law")  # powerlaw
    plot.legend(loc="lower left")
    plot.show()


    R, p = results.distribution_compare('power_law', 'exponential')
    print(f"Loglikelihood ratio: {R}")
    print(f"Statistical significance: {p}")

    R, p = results.distribution_compare('power_law', 'stretched_exponential')
    print(f"Loglikelihood ratio: {R}")
    print(f"Statistical significance: {p}")

    R, p = results.distribution_compare('power_law', 'truncated_power_law')
    print(f"Loglikelihood ratio: {R}")
    print(f"Statistical significance: {p}")

    R, p = results.distribution_compare('power_law', 'lognormal_positive')
    print(f"Loglikelihood ratio: {R}")
    print(f"Statistical significance: {p}")
