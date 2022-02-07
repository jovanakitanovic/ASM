# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from functools import reduce
import networkx as nx
from collections import Counter
import powerlaw
import matplotlib.pyplot as plot

import functions

def reading_data_submissions(data):
    submissions_data_0 = pd.read_csv(
        r"submissions_2008_asm\csv-0.csv")
    subreddit_id_0 = submissions_data_0["subreddit_id"].unique()
    subreddit_author_0 = submissions_data_0[["author", "subreddit_id"]]

    submissions_data_1 = pd.read_csv(
        r"submissions_2008_asm\csv-1.csv")
    subreddit_id_1 = submissions_data_1["subreddit_id"].unique()
    subreddit_author_1 = submissions_data_1[["author", "subreddit_id"]]

    submissions_data_2 = pd.read_csv(
        r"submissions_2008_asm\csv-2.csv")
    subreddit_id_2 = submissions_data_2["subreddit_id"].unique()
    subreddit_author_2 = submissions_data_2[["author", "subreddit_id"]]

    submissions_data_3 = pd.read_csv(
        r"submissions_2008_asm\csv-3.csv")
    subreddit_id_3 = submissions_data_3["subreddit_id"].unique()
    subreddit_author_3 = submissions_data_3[["author", "subreddit_id"]]

    submissions_data_4 = pd.read_csv(
        r"submissions_2008_asm\csv-4.csv")
    subreddit_id_4 = submissions_data_4["subreddit_id"].unique()
    subreddit_author_4 = submissions_data_4[["author", "subreddit_id"]]

    submissions_data_5 = pd.read_csv(
        r"submissions_2008_asm\csv-5.csv")
    subreddit_id_5 = submissions_data_5["subreddit_id"].unique()
    subreddit_author_5 = submissions_data_5[["author", "subreddit_id"]]

    submissions_data_6 = pd.read_csv(
        r"submissions_2008_asm\csv-6.csv")
    subreddit_id_6 = submissions_data_6["subreddit_id"].unique()
    subreddit_author_6 = submissions_data_6[["author", "subreddit_id"]]

    submissions_data_7 = pd.read_csv(
        r"submissions_2008_asm\csv-7.csv")
    subreddit_id_7 = submissions_data_7["subreddit_id"].unique()
    subreddit_author_7 = submissions_data_7[["author", "subreddit_id"]]

    submissions_data_8 = pd.read_csv(
        r"submissions_2008_asm\csv-8.csv")
    subreddit_id_8 = submissions_data_8["subreddit_id"].unique()
    subreddit_author_8 = submissions_data_8[["author", "subreddit_id"]]

    submissions_data_9 = pd.read_csv(
        r"submissions_2008_asm\csv-9.csv")
    subreddit_id_9 = submissions_data_9["subreddit_id"].unique()
    subreddit_author_9 = submissions_data_9[["author", "subreddit_id"]]

    submissions_data_10 = pd.read_csv(
        r"submissions_2008_asm\csv-10.csv")
    subreddit_id_10 = submissions_data_10["subreddit_id"].unique()
    subreddit_author_10 = submissions_data_10[["author", "subreddit_id"]]

    submissions_data_11 = pd.read_csv(
        r"submissions_2008_asm\csv-11.csv")
    subreddit_id_11 = submissions_data_11["subreddit_id"].unique()
    subreddit_author_11 = submissions_data_11[["author", "subreddit_id"]]

    # ----------all nodes that should be in net /begin/-------------------------------------------------------------------

    subreddit_id_all = reduce(np.union1d,
                              (subreddit_id_0, subreddit_id_1, subreddit_id_2, subreddit_id_3, subreddit_id_4,
                               subreddit_id_5, subreddit_id_6, subreddit_id_7, subreddit_id_8, subreddit_id_9,
                               subreddit_id_10, subreddit_id_11,data[1]))

    submissions_data_all = [submissions_data_0,submissions_data_1,submissions_data_2,submissions_data_3,submissions_data_4,submissions_data_5,submissions_data_6,submissions_data_7,submissions_data_8,submissions_data_9,submissions_data_10,submissions_data_11]
    submissions_data_all = pd.concat(submissions_data_all)

    """arr_class_9=pd.read_csv(
        r"klasa9.csv")
    arr1=[]
    for i in arr_class_9["data"]:
        print(i)
        arr1.append(i)

    num=0

    for elem in submissions_data_all.iterrows():
        if elem[1]["subreddit_id"] in arr1:
            num+=1
            print(elem[1]["subreddit"],9)
            arr1.remove(elem[1]["subreddit_id"])"""




    print(f"ukupan broj različitih subredita je {len(subreddit_id_all)}")

    # ----------all nodes that should be in net /end/-------------------------------------------------------------------

    # ----------all authors that modified subreddits, edges will be created from this data /begin/--------------------------------

    subreddit_participant = [subreddit_author_0, subreddit_author_1, subreddit_author_2, subreddit_author_3,
                             subreddit_author_4, subreddit_author_5, subreddit_author_6, subreddit_author_7,
                             subreddit_author_8, subreddit_author_9, subreddit_author_10, subreddit_author_11,
                             data[0]]
    subreddit_participant = pd.concat(subreddit_participant)
    #print(f"BROJ UKUPAN UČESNIKA NEFILTRIRANO {len(subreddit_participant)}")

    filtered_data=subreddit_participant['author']!="[deleted]"
    subreddit_participant=subreddit_participant[filtered_data]
    """!!!"""
    #functions.subreddit_with_most_participants(subreddit_participant)
    """!!!"""
    subreddit_participant_withot_comments = [subreddit_author_0, subreddit_author_1, subreddit_author_2, subreddit_author_3,
                             subreddit_author_4, subreddit_author_5, subreddit_author_6, subreddit_author_7,
                             subreddit_author_8, subreddit_author_9, subreddit_author_10, subreddit_author_11]
    """!!!"""
    #author_with_most_submissions=functions.author_with_most_submissions(subreddit_participant_withot_comments);

    #functions.average_num_author_per_subreddit(subreddit_participant,len(subreddit_id_all))
    #functions.piersons_correlation(data[2],author_with_most_submissions)

    # ----------all authors that modified subreddits, edges will be created from this data /end/--------------------------------

    subreddit_participant = subreddit_participant.groupby(["author", "subreddit_id"])

    #functions.users_active_on_most_subreddits(subreddit_participant)
    """!!!"""

    print(len(subreddit_participant))

    data_network_node=create_net(subreddit_id_all, subreddit_participant.groups)
    return data_network_node

def reading_data_comments():
    comment_data_0 = pd.read_csv(
        r"comments_2008_asm\csv-0.csv")
    subreddit_id_0 = comment_data_0["subreddit_id"].unique()
    subreddit_comment_author_0 = comment_data_0[["author", "subreddit_id"]]

    comment_data_1 = pd.read_csv(
        r"comments_2008_asm\csv-1.csv")
    subreddit_id_1 = comment_data_1["subreddit_id"].unique()
    subreddit_comment_author_1 = comment_data_1[["author", "subreddit_id"]]

    comment_data_2 = pd.read_csv(
        r"comments_2008_asm\csv-2.csv")
    subreddit_id_2 = comment_data_2["subreddit_id"].unique()
    subreddit_comment_author_2 = comment_data_2[["author", "subreddit_id"]]

    comment_data_3 = pd.read_csv(
        r"comments_2008_asm\csv-3.csv")
    subreddit_id_3 = comment_data_3["subreddit_id"].unique()
    subreddit_comment_author_3 = comment_data_3[["author", "subreddit_id"]]

    comment_data_4 = pd.read_csv(
        r"comments_2008_asm\csv-4.csv")
    subreddit_id_4 = comment_data_4["subreddit_id"].unique()
    subreddit_comment_author_4 = comment_data_4[["author", "subreddit_id"]]

    comment_data_5 = pd.read_csv(
        r"comments_2008_asm\csv-5.csv")
    subreddit_id_5 = comment_data_5["subreddit_id"].unique()
    subreddit_comment_author_5 = comment_data_5[["author", "subreddit_id"]]

    comment_data_6 = pd.read_csv(
        r"comments_2008_asm\csv-6.csv")
    subreddit_id_6 = comment_data_6["subreddit_id"].unique()
    subreddit_comment_author_6 = comment_data_6[["author", "subreddit_id"]]

    comment_data_7 = pd.read_csv(
        r"comments_2008_asm\csv-7.csv")
    subreddit_id_7 = comment_data_7["subreddit_id"].unique()
    subreddit_comment_author_7 = comment_data_7[["author", "subreddit_id"]]

    comment_data_8 = pd.read_csv(
        r"comments_2008_asm\csv-8.csv")
    subreddit_id_8= comment_data_8["subreddit_id"].unique()
    subreddit_comment_author_8 = comment_data_8[["author", "subreddit_id"]]

    comment_data_9 = pd.read_csv(
        r"comments_2008_asm\csv-9.csv")
    subreddit_id_9 = comment_data_9["subreddit_id"].unique()
    subreddit_comment_author_9 = comment_data_9[["author", "subreddit_id"]]

    comment_data_10 = pd.read_csv(
        r"comments_2008_asm\csv-10.csv")
    subreddit_id_10 = comment_data_10["subreddit_id"].unique()
    subreddit_comment_author_10 = comment_data_10[["author", "subreddit_id"]]

    comment_data_11 = pd.read_csv(
        r"comments_2008_asm\csv-11.csv")
    subreddit_id_11 = comment_data_11["subreddit_id"].unique()
    subreddit_comment_author_11 = comment_data_11[["author", "subreddit_id"]]

    # ----------all nodes that should be in net /begin/-------------------------------------------------------------------

    subreddit_id_all = reduce(np.union1d,
                              (subreddit_id_0, subreddit_id_1, subreddit_id_2, subreddit_id_3, subreddit_id_4,
                               subreddit_id_5, subreddit_id_6, subreddit_id_7, subreddit_id_8, subreddit_id_9,
                               subreddit_id_10, subreddit_id_11))

    print(subreddit_id_all)

    print(f"ukupan broj različitih subredita kod komentara je {len(subreddit_id_all)}")

    # ----------all nodes that should be in net /end/-------------------------------------------------------------------

    # ----------all authors that commented on subreddits, data will be concatenated in reading_data_submissions()---------
    # ----------and used to create edges /begin/---------------------------------------------------------------------------

    subreddit_comment_author = [subreddit_comment_author_0, subreddit_comment_author_1, subreddit_comment_author_2,
                                subreddit_comment_author_3,
                                subreddit_comment_author_4, subreddit_comment_author_5, subreddit_comment_author_6,
                                subreddit_comment_author_7,
                                subreddit_comment_author_8, subreddit_comment_author_9, subreddit_comment_author_10,
                                subreddit_comment_author_11]
    subreddit_comment_author = pd.concat(subreddit_comment_author)
    filtered_data=subreddit_comment_author['author']!="[deleted]"
    subreddit_comment_author=subreddit_comment_author[filtered_data]

    #functions.subreddit_with_most_comments(subreddit_comment_author)
    author_with_most_comments=None
    #author_with_most_comments=functions.author_with_most_comments(subreddit_comment_author)


    # ----------all authors that commented on subreddits, data will be concatenated in reading_data_submissions()---------
    # ----------and used to create edges /end/---------------------------------------------------------------------------

    return [subreddit_comment_author,subreddit_id_all,author_with_most_comments];


def create_net(nodes, edge_data):
    network = nx.Graph()
    network.add_nodes_from(nodes)

    #pos = nx.circular_layout(network)
    #nx.draw_networkx(network, pos)
    # plot.show()

    current_elem=""
    array_of_nodes=[]

    for elem, group in edge_data:
        #print(elem,group)
        if(current_elem!=elem ):
            if(len(array_of_nodes)>0):
                for i in range(0,len(array_of_nodes)):
                    for j in range(i+1,len(array_of_nodes)):
                        #print(f"STAVLJENO {array_of_nodes[i]} ,{array_of_nodes[j]},{elem}")
                        if(array_of_nodes[i],array_of_nodes[j]) in network.edges:
                            network.edges[array_of_nodes[i],array_of_nodes[j]]['weight']+=1
                        else:
                            network.add_edge(array_of_nodes[i],array_of_nodes[j], weight=1)

            current_elem=elem
            array_of_nodes=[]
            array_of_nodes.append(group)
        else:
            array_of_nodes.append(group)


    nx.write_gml(network,"SNet.gml")

    """!!!"""
    #randomGraphSameSize=nx.gnm_random_graph(network.number_of_nodes(),network.number_of_edges())
    #nx.write_gml(randomGraphSameSize,"randomGraphSameSize.gml")

    #randomGraphErdosRenyi=nx.erdos_renyi_graph(network.number_of_nodes(),0.01)
    #nx.write_gml(randomGraphErdosRenyi,"randomGraphErdosRenyi.gml")

    #functions.network_clasterization(network,randomGraphSameSize,randomGraphErdosRenyi)

    """!!!"""

    print("network created!")

    return network


if __name__ == '__main__':
    data = reading_data_comments()
    network = reading_data_submissions(data)

    print(nx.has_bridges(network))
    print(list(nx.bridges(network)))


    """!!!"""
    #functions.katz_centrality(network,1,1)
    #functions.katz_centrality(network,2,0.5)
    #functions.katz_centrality(network,0.5,0.1)
    #functions.katz_centrality(network,3,1)


    #power_law_distribution(network,False)

    #functions.centrality_analysis(network)
    #functions.plot_dendogram(network)


    #functions.dominant_component(network)

    #functions.network_assortativity(network)
    #functions.eigen_centrality(network)
    """!!!"""

    """OBJAŠNJENJE KREIRANJA MREŽE:
    iz fajlova o objavama i komentarima su uzete sve vrednosti subreddit_id-jeva, zatim su one isfiltrirane tako da u nizu id-jeva ostane samo po jedno pojavljivanje istog.
    Tako isfiltrirane vrednosti predstavljaju čvorove mreže
    iz fajlova o objavama i komentarima su uzete vrednosti autora i subreddit_id, po te dve vdednosti su i grupiasani tako da ostane samo po jedno pojavljivanje kombijacije 
    autor subredit. U ptlji za dodavanje grane se proverava na koliko je različitih subredita autor učestvovao a zatim se između njih pravi kompetan graf. (svaki sa svakim se povezuje)"""

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
