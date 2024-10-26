######################################
# Class and functions to handle verification analysis
######################################

import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import numpy as np
import seaborn as sns

from matplotlib.gridspec import GridSpec

YEAR = 2021

def plotVerificationMetrics(ax=plt):
    # Fusion Target Metrics
    ax.scatter([.01],[.85],marker='+',label="Fusion Target Metric (Phase 1)",color='green')

    ax.scatter([.01],[.7],marker='+',label="Face Target Metric (Phase 1)",color='blue')

def plotVerificationMetricsFaceRestricted(ax=plt):
    # Fusion Target Metrics
    ax.scatter([.01],[.5],marker='+',label="Fusion Target Metric (Phase 1)",color='green')

def metrics_at_thersholds(dataframe):
    tar = dataframe['tar']
    far = dataframe['far']
    metrics = []
    for i in [1e-1,1e-2,1e-3,1e-4,1e-5]:
        v1 = np.where(far == min(far, key=lambda x:(x-i) and x > i))
        v2 = np.where(far == min(far, key=lambda x:(i - x) and x < i))
        val = (tar[v1[0][-1]] +  tar[v2[0][-1]])/2
        metrics.append(val)
    return metrics

def roc_curve(mask,scores):
    '''
    A new version of this function that assumes failures come first.
    '''
    #data = np.array([mask,scores])
    scores = -scores
    match = scores[mask]
    nonmatch = scores[~mask]
    
    m = len(match)
    n = len(nonmatch)
    both = np.concatenate([match,nonmatch])
    matches = np.array(len(match)*[1]+len(nonmatch)*[0])
    nonmatches = np.array(len(match)*[0]+len(nonmatch)*[1])
    order = both.argsort()
    scores = both[order]
    matches = matches[order]
    nonmatches = nonmatches[order]
    tar = matches.cumsum()/float(m)
    far = nonmatches.cumsum()/float(n)
    keep = np.ones(len(tar),dtype=np.bool)
    keep[1:-1][(far[:-2] == far[1:-1]) & (far[2:] == far[1:-1])] = False
    keep[1:-1][(tar[:-2] == tar[1:-1]) & (tar[2:] == tar[1:-1])] = False
    scores = scores[keep]
    tar = tar[keep]
    far = far[keep]
    rows = np.array([scores,1.0-tar,far,1.0-far,tar]).T

    return far, tar, -scores




class VerificationResults(object):
    '''
    A data structure to manage a verifcation score matrix.

    This keeps results in the same place.
    '''

    def __init__(self, score_matrix, probe_df, gallery_df,algorithm="Unknown",label="Unknown",color='gray',linewidth=3,linestyle='solid'):
        ''' Create a verification results object to track scores. '''
        self.score_matrix = score_matrix
        self.probe_df = probe_df
        self.gallery_df = gallery_df
        self.algorithm = " ".join(algorithm.split("_"))
        self.label = label
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.generateSubIds()
        self.generateMask()
        self._roc_auc = None
        self._roc_df = None
        self._thresholds_df = None
        



    def generateSubIds(self):
        '''Called on initialization to generate a lists of probe and gallery subject ids'''
        self.probe_ids = self.probe_df['subject_id'].to_numpy()
        self.gallery_ids = self.gallery_df['subject_id'].to_numpy()

        assert len(self.probe_ids) == self.score_matrix.shape[0]
        assert len(self.gallery_ids) == self.score_matrix.shape[1]


    def generateMask(self):
        ''' Called on initialization to generate a mask matrix. '''
        self.mask_matrix = self.probe_ids.reshape(-1,1) == self.gallery_ids.reshape(1,-1)

        assert self.mask_matrix.shape == self.score_matrix.shape

    
    def createOldRoc(self):
        '''
        Create a roc curve data from lists of match scores and nonmatch scores.
        '''
        if self._roc_df is not None:
            return self._roc_df,self._roc_auc

        scores = self.score_matrix.flatten().copy()
        mask = self.mask_matrix.flatten()

        # Nan numbers should be treated as failures or failure to enroll. (lowest score)
        nan_mask = np.isnan(scores)
        #print(len(scores.flatten()),nan_mask.sum())
        if (~nan_mask).sum() > 0:
            scores[nan_mask] = scores.flatten()[~nan_mask.flatten()].min()
        else:
            scores[nan_mask] = 0.0
        #print(len(scores.flatten()),nan_mask.sum(), np.isnan(scores).sum())


        if len(scores) != 0:
            fpr, tpr, threshold = roc_curve(mask, scores)
            #fpr, tpr, threshold = metrics.roc_curve(mask, scores,pos_label=1)
            auc = metrics.roc_auc_score(mask,scores)
            data = list(zip(list(range(len(fpr))),threshold,tpr,fpr))
            df = pd.DataFrame(data,columns=('step','match_score','tar','far'))
            
            self._roc_df = df
            self._roc_auc = auc
        else:
            self._roc_df = None
            self._roc_auc = 0.0

        return self._roc_df,self._roc_auc
        
    def createRoc(self, far_range=(0,-5), n_points=2000):
        '''
        Create a roc curve data from lists of match scores and nonmatch scores.
        '''
        scores = self.score_matrix.flatten().copy()
        mask = self.mask_matrix.flatten()

        # Nan numbers should be treated as failures or failure to enroll. (lowest score)
        nan_mask = np.isnan(scores)

        if (~nan_mask).sum() > 0:
            scores[nan_mask] = scores.flatten()[~nan_mask.flatten()].min()
        else:
            scores[nan_mask] = 0.0

        # Process the negitives of the scores.
        match = -scores[mask]
        nonmatch = -scores[~mask]
        match.sort()
        nonmatch.sort()

        # Compute the curve
        target_far = 10**np.linspace(far_range[0],far_range[1],n_points+1)
        nm_select = len(nonmatch)*target_far
        nm_select = np.array(np.floor(nm_select)-1,dtype=np.int)
        nm_select = np.maximum(nm_select,1)
        nm_scores = nonmatch[::-1][-nm_select]
        true_far = (nm_select+1.0)/len(nonmatch)
        m_select = np.searchsorted(match,nm_scores,side='left')
        m_scores = match[np.minimum(m_select,len(match)-1)]
        tar = m_select/len(match)

        # Generate results table
        tmp = list(zip(target_far,true_far,nm_select,m_select,-nm_scores,-m_scores,tar))
        roc = pd.DataFrame(tmp,columns=['target_far','true_far','nm_select','m_select','nm_scores','m_scores','tar'])
        
        # Compute the auc
        bin_sizes =  roc.true_far.to_numpy()[:-1] - roc.true_far.to_numpy()[1:]
        bin_ave = 0.5*roc.tar.to_numpy()[:-1] + 0.5*roc.tar.to_numpy()[1:]
        auc = (bin_sizes*bin_ave).sum()
        
        return roc, auc
        

    def thresholds(self):
        '''
        Compute a set of FAR thresholds from these scores.  This is only computed for valid (real) scores.
        '''
        if self._thresholds_df is not None:
            return self._thresholds_df

        scores = self.score_matrix.flatten()
        mask = ~self.mask_matrix.flatten()
        scores = scores[mask]

        # limit the computation to real scores
        scores = scores[~np.isnan(scores)]

        if len(scores) == 0:
            scores = np.zeros(1000000,dtype=np.float64)
        
        data = []
        index = []
        scores.sort()
        n = len(scores)
        for tmp in [100,1000,10000,100000,1000000]:
            index.append(tmp)
            far = 1./tmp
            idx = min(int(round((1-far)*n)),len(scores)-1)
            score = scores[idx]
            
            rank_med = int(np.round(far*n))
            rank_low = int(np.floor(opt.fmin(lambda k: (stats.binom.cdf(k,n,far)-0.025)**2, n*far,disp=False)[0]))
            rank_high = int(np.ceil(opt.fmin(lambda k: (stats.binom.cdf(k,n,far)-0.975)**2, n*far,disp=False)[0]))

            rank_low = max(rank_low,0)
            rank_low = min(rank_low,len(scores) - 1)
            rank_high = max(rank_high,0)
            rank_high = min(rank_high,len(scores) - 1)

            score_low = scores[::-1][rank_low]
            score_med = scores[::-1][rank_med]
            score_high = scores[::-1][rank_high]

            data.append([idx,n,far,score,rank_low,rank_med,rank_high,score_low,score_med,score_high,rank_low/n,rank_high/n])


        df = pd.DataFrame(data,columns=['index','total','far','score','low_95','med','high_95','score_low','score_med','score_high','ci_low','ci_high'],index=index)
        
        self._thresholds_df = df
        return self._thresholds_df

    def plotThresholds(self, ax):
        ''' Show the thresholds at different levels of far. '''
        ymin,ymax = ax.get_ylim()
        xmin,xmax = ax.get_ylim()

        thresholds = self.thresholds()
        for row in thresholds.iloc:
            plt.vlines(row['score'],ymin-0.05*ymax,ymax+0.05*ymax,colors='gray',linestyles='dotted')
            ax.text(row['score'],ymax,"FAR@%0.5f = %+0.4f"%(1.0/float(row.name),row['score']),verticalalignment='top',horizontalalignment='left',fontsize=8,rotation=90,color='gray')


    def plotRoc(self,axis=plt,label=None, algorithm=False,subjects=False,linestyle=None, linewidth=None, color=None, n_points=2000, **kwargs):
        df,auc = self.createRoc(n_points=n_points)

        # Cutoff if there is less that 10 nonmatch scores
        df = df.where(df.nm_select >= 10)

        if label == None:
            label="{} n={} auc={:.3f} tar@0.01far={:.3f}".format(self.label,len(self.probe_df),auc,
                                                              df[df['true_far'] >= 0.01].iloc[0]['tar'])

            if algorithm:
                label = self.algorithm + " - " + label 


        if linestyle is None and len(self.probe_df) < 200:
            linestyle=(0, (7, 10))
        elif linestyle is None:
            linestyle=self.linestyle

        if linewidth is None and len(self.probe_df) < 200:
            linewidth=1
        else:
            linewidth=self.linewidth

        if color is None:
            color=self.color

        axis.plot(df['true_far'],df['tar'],label=label,linestyle=linestyle,linewidth=linewidth,color=color,**kwargs)
        
        try:
            axis.xlim(1e-5,1e-0)
            axis.ylim(-0.05,1.05)
            axis.xlabel('False Accept Rate')
            axis.ylabel('True Accept Rate')
            axis.xscale('log')
        except:
            axis.set_xlim(1e-5,1e-0)
            axis.set_ylim(-0.05,1.05)
            axis.set_xlabel('False Accept Rate')
            axis.set_ylabel('True Accept Rate')
            axis.set_xscale('log')

    def scoreRange(self):
        scores = self.score_matrix.flatten()
        scores = scores[~np.isnan(scores)]
        try:
            return scores.min(),scores.max()
        except:
            return -1.0,1.0


    def matchScores(self):
        scores = self.score_matrix.flatten()
        mask = self.mask_matrix.flatten()
        scores = scores[mask]
        assert len(scores) == mask.sum()
        return scores


    def nonMatchScores(self):
        scores = self.score_matrix.flatten()
        mask = ~self.mask_matrix.flatten()
        scores = scores[mask]
        assert len(scores) == mask.sum()
        return scores


    def plotNotmatchHistogram(self,bins=100,color='blue',axis=None):
        scores = self.nonMatchScores()
        scores = scores[~np.isnan(scores)]
        return plt.hist(scores,bins=bins,color=color,density=False)
    

    def plotMatchHistogram(self,bins=100,color='green',axis=None):
        scores = self.matchScores()
        scores = scores[~np.isnan(scores)]
        return plt.hist(scores,bins=bins,color=color,density=False)

    def plotProbeSamples(self,ax,N=30):

        probe_select = np.arange(self.score_matrix.shape[0])
        np.random.shuffle(probe_select)
        probe_select = probe_select[:N]
        probe_select.sort()

        labels = []
        match_scores = []
        nonmatch_scores = []
        for i in probe_select:
            labels.append(self.probe_df['entry_id'][i])

            scores = self.score_matrix[i,:][~self.mask_matrix[i,:]]
            scores = scores[~np.isnan(scores)]
            nonmatch_scores.append(scores)

            scores = self.score_matrix[i,:][self.mask_matrix[i,:]]
            scores = scores[~np.isnan(scores)]
            match_scores.append(scores)

        #ax.boxplot(body_scores[:,:50],vert=False,showfliers=True,whis=(0, 100),labels=gallery['subject_id'][:50])
        ax.boxplot(nonmatch_scores,vert=False,showfliers=True,whis=(0, 100),labels=labels)
        for i,scores in enumerate(match_scores):
            ax.scatter(scores,[i+1]*len(scores),25,color='#00AA00AA')

    def plotGallerySamples(self,ax,N=30):

        gal_select = np.arange(self.score_matrix.shape[1])
        np.random.shuffle(gal_select)
        gal_select = gal_select[:40]
        gal_select.sort()

        labels = []
        match_scores = []
        nonmatch_scores = []
        for i in gal_select:
            labels.append(self.gallery_df['subject_id'][i])

            scores = self.score_matrix[:,i][~self.mask_matrix[:,i]]
            scores = scores[~np.isnan(scores)]
            nonmatch_scores.append(scores)

            scores = self.score_matrix[:,i][self.mask_matrix[:,i]]
            scores = scores[~np.isnan(scores)]
            match_scores.append(scores)

        ax.boxplot(nonmatch_scores,vert=False,showfliers=True,whis=(0, 100),labels=labels)
        for i,scores in enumerate(match_scores):
            ax.scatter(scores,[i+1]*len(scores),25,color='#00AA0033')



    def addAlgWatermark(self,ax):
        x = 0.995*ax.get_xlim()[0] + 0.005*ax.get_xlim()[1]
        y = 0.005*ax.get_ylim()[0] + 0.995*ax.get_ylim()[1]
        ax.text(x,y,self.algorithm,verticalalignment='top',horizontalalignment='left',color='lightgray')

    def filterProbesSelection(self, filter_func, algorithm=None, label=None, color=None, linewidth=None,
                              linestyle=None):
        '''
        Filter function takes the probe dataframe as an argument and returns a boolean list of probes to keep.

        @return: a new VerificationResult object with just those results.
        '''
        if algorithm == None:
            algorithm = self.algorithm

        try:
            selection = filter_func(self.probe_df)
        except:
            selection = filter_func

        if color is None:
            color = self.color

        if linewidth is None:
            linewidth = self.linewidth

        if linewidth is None:
            linestyle = self.linestyle

        if label is None:
            label = self.label

        # result = VerifcationResults(self.score_matrix[selection,:],self.probe_df[selection],self.gallery_df,algorithm=algorithm,label=label,linewidth=linewidth,color=color,linestyle=linestyle)
        return selection

    def filterProbes(self,filter_func,algorithm=None,label=None,color=None,linewidth=None,linestyle=None):
        '''
        Filter function takes the probe dataframe as an argument and returns a boolean list of probes to keep.

        @return: a new VerificationResult object with just those results.
        '''
        if algorithm == None:
            algorithm = self.algorithm

        try:
            selection = filter_func(self.probe_df)
        except:
            selection = filter_func

        if color is None:
            color = self.color

        if linewidth is None:
            linewidth = self.linewidth

        if linewidth is None:
            linestyle = self.linestyle

        if label is None:
            label = self.label 

        result = VerificationResults(self.score_matrix[selection,:],self.probe_df[selection],self.gallery_df,algorithm=algorithm,label=label,linewidth=linewidth,color=color,linestyle=linestyle)
        return result

    def filterGallery(self,filter_func,algorithm=None,label="Unknown",color=None,linewidth=None,linestyle=None):
        '''
        Filter function takes the probe dataframe as an argument and returns a boolean list of probes to keep.

        @return: a new VerificationResult object with just those results.
        '''
        if algorithm == None:
            algorithm = self.algorithm

        try:
            selection = filter_func(self.probe_df)
        except:
            selection = filter_func

        if color is None:
            color = self.color

        if linewidth is None:
            linewidth = self.linewidth

        if linewidth is None:
            linestyle = self.linestyle

        result = VerificationResults(self.score_matrix[:,selection],self.probe_df,self.gallery_df[selection],algorithm=algorithm,label=label,linewidth=linewidth,color=color,linestyle=linestyle)
        return result


    def impostorConfusionMatrix(self, threshold):
        '''Count matches between impostors based on a score threshold.'''

        gallery_ids = list(self.gallery_df['subject_id'])

        m1 = self.score_matrix >= threshold
        m2 = ~self.mask_matrix
        counts = 1*(m1 & m2)

        conf = {}
        for record,row in zip(self.probe_df.to_dict(orient='records'),counts):
            #print(record)
            subject_id = record['subject_id']
            if subject_id not in conf:
                conf[subject_id] = row
            else:
                conf[subject_id] += row
            #print(row,record)
        probe_ids = list(conf.keys())
        probe_ids.sort()

        mat = [conf[pid] for pid in probe_ids]
        mat = np.array(mat,dtype=np.int64)

        return probe_ids,gallery_ids,mat

    def plotImpostorConfusionMatrix(self,threshold):

        probe_ids,gallery_ids,conf_mat = self.impostorConfusionMatrix(threshold)

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005


        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.05, height]

        fig = plt.figure(figsize=(20, 10))


        ax = fig.add_axes(rect_scatter)
        mat_mappable = ax.imshow(conf_mat,cmap='YlOrRd',aspect='auto')

        peaks = conf_mat.sum(axis=0)
        tmp = peaks.copy()
        tmp.sort()
        tmp = tmp[::-1]
        n = len(tmp)
        thresh = tmp[20]
        mask = peaks >= thresh
        loc= np.nonzero(mask)[0]
        lab = np.array(gallery_ids)[loc]
        #print(mask,np.nonzero(mask))
        #print(loc)
        #print(lab)
        plt.xticks(loc,lab,rotation = 90)

        peaks = conf_mat.sum(axis=1)
        tmp = peaks.copy()
        tmp.sort()
        tmp = tmp[::-1]
        n = len(tmp)
        thresh = tmp[12]
        mask = peaks >= thresh
        loc= np.nonzero(mask)[0]
        lab = np.array(probe_ids)[loc]
        #print(mask,np.nonzero(mask))
        #print(loc)
        #print(lab)
        plt.yticks(loc,lab,rotation = 0)


        plt.xlabel('Gallery Entries')
        plt.ylabel('Probe Entries grouped by Subject Id')

        ax_countx = fig.add_axes(rect_histx, sharex=ax)
        ax_countx.bar(np.arange(len(gallery_ids)),conf_mat.sum(axis=0))
        ax_countx.tick_params(axis="x", labelbottom=False)

        ax_county = fig.add_axes(rect_histy, sharey=ax)
        ax_county.barh(np.arange(len(probe_ids)),conf_mat.sum(axis=1))
        ax_county.tick_params(axis="y", labelleft=False)

        plt.colorbar(mat_mappable,ax=ax_county)



    def splitProbesFromImpostorTail(self, verbose = True):
        '''
        Split the scores based based on probes that are contributing to 
        the impostor tail.  This should artifically boost accuracy of the 
        results.
        '''
        THRESHOLD_SECTION = 100 # Eliminate the 1/100 FAR tail
        PROBE_IMPOSTOR_REDUCTION = 0.9  # Try to remove probes in a greedy manor to get rid of 90% of the tail.


        face_thresh = self.thresholds().loc[THRESHOLD_SECTION,'score']
        face_far = self.thresholds().loc[THRESHOLD_SECTION,'far']

        if verbose:
            print("Included Score:",face_thresh, 'FAR Rate:', face_far)
        
        m2 = ~self.mask_matrix
        m1 = self.score_matrix > face_thresh

        tmp = (m1 & m2).sum(axis=1)

        thresh = tmp.max()
        while thresh > 0:    
            remove = tmp[tmp>thresh].sum()
            if remove/tmp.sum() > PROBE_IMPOSTOR_REDUCTION:
                break
            thresh -= 1

        mask = tmp>thresh

        removed_probes = mask.sum()
        tail_coverage = remove/tmp.sum()
        total_probes = mask.shape[0]
        if verbose:
            print('There are %d of %d (%0.2f%%) probe enrtries that account for %0.2f%% of the impostor tail.'%(removed_probes,total_probes, 100.0*removed_probes/total_probes,100.0*tail_coverage))
        
        hard_imp_removed = self.filterProbes(~mask,label='Face - Hard Impostors Removed')
        hard_imp_only = self.filterProbes(mask,label='Face - Hard Impostors Only')
        
        return hard_imp_removed, hard_imp_only



    def splitGalleryFromImpostorTail(self, verbose = True):
        '''
        Split the scores based based on gallery entries that are contributing to 
        the impostor tail.  This should artifically boost accuracy of the 
        results.
        '''
        THRESHOLD_SECTION = 100
        PROBE_IMPOSTOR_REDUCTION = 0.9 # Eliminate gallery entries causing 90% of the tail

        face_thresh = self.thresholds().loc[THRESHOLD_SECTION,'score']
        face_far = self.thresholds().loc[THRESHOLD_SECTION,'far']
        if verbose:
            print("Included Score:",face_thresh, 'FAR Rate:', face_far)
        m2 = ~self.mask_matrix
        m1 = self.score_matrix > face_thresh

        tmp = (m1 & m2).sum(axis=0)

        thresh = tmp.max()
        while thresh > 0:    
            remove = tmp[tmp>thresh].sum()
            #print(thresh,remove, tmp.sum(), remove/tmp.sum(),(tmp>thresh).sum())
            if remove/tmp.sum() > PROBE_IMPOSTOR_REDUCTION:
                break
            thresh -= 1

        mask = tmp>thresh

        removed_probes = mask.sum()
        tail_coverage = remove/tmp.sum()
        total_probes = mask.shape[0]
        if verbose:
            print('There are %d of %d (%0.2f%%) gallery enrtries that account for %0.2f%% of the impostor tail.'%(removed_probes,total_probes, 100.0*removed_probes/total_probes,100.0*tail_coverage))

        hard_imp_removed = self.filterGallery(~mask,label='Face - Hard Impostors Removed')
        hard_imp_only = self.filterGallery(mask,label='Face - Hard Impostors Only')

        return hard_imp_removed, hard_imp_only


    def plotDemographics(self):
        demographics_df = self.probe_df.drop_duplicates(subset=['subject_id']).copy()

        demographics_df.loc[demographics_df['year_of_birth']=='UNK','year_of_birth'] = 'nan'
        demographics_df['year_of_birth'] = np.array(demographics_df['year_of_birth'],dtype=np.float32)

        demographics_df.loc[demographics_df['height_inches']=='UNK','height_inches'] = 'nan'
        demographics_df['height_inches'] = np.array(demographics_df['height_inches'],dtype=np.float32)

        demographics_df.loc[demographics_df['weight_lbs']=='UNK','weight_lbs'] = 'nan'
        demographics_df['weight_lbs'] = np.array(demographics_df['weight_lbs'],dtype=np.float32)

        demographics_df['bmi'] = 703 * demographics_df['weight_lbs'] / (demographics_df['height_inches'] ** 2)
        demographics_df.race = demographics_df.race.str.title()
        demographics_df.sex = demographics_df.sex.str.title()


        fig = plt.figure(constrained_layout=True, figsize=(10,10))
        fig.suptitle(f'Demographics')

        gs = GridSpec(nrows=4, ncols=2, figure=fig)

        ax1 = fig.add_subplot(gs[0,0])
        sex_df = demographics_df.groupby('sex').count().reset_index()
        ax1.pie(x=sex_df['subject_id'],labels=sex_df['sex'].str.capitalize())
        ax1.set_title('Subject Sex Distribution')

        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
        ax2 = fig.add_subplot(gs[0,1])
        race_df = demographics_df.groupby('race').count().reset_index()
        ax2.pie(x=race_df['subject_id'],labels=race_df['race'].str.title())
        ax2.set_title('Subject Race Distribution')


        ax3 = fig.add_subplot(gs[1,0])
        ax3.hist(YEAR-demographics_df['year_of_birth'],bins=np.arange(0,100,5))
        ax3.set(xlabel='Age in 5 year bins',
                ylabel='Subject Count',
                title='Subject Age Distribution')


        ax4 = fig.add_subplot(gs[2,0])
        ax4.hist(demographics_df['height_inches'])
        ax4.set(xlabel='Height in Inches',
                ylabel='Subject Count',
                title="Subject Height Distribution")

        ax5 = fig.add_subplot(gs[3,0])
        ax5.hist(demographics_df['weight_lbs'])
        ax5.set(xlabel='Weight in Lbs',
                ylabel='Subject Count',
                title="Subject Weight Distribution")

        ax6 = fig.add_subplot(gs[1:,-1])
        g = sns.barplot(data=demographics_df\
                            .groupby(['race','sex'])\
                            .agg(counts=('sex','count'))\
                            .reset_index(),
                        x='race',
                        y='counts',
                        hue='sex',
                        ax=ax6
                    )

        for p in ax6.patches:
            ax6.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                        textcoords='offset points')

        ax6.set(xlabel='Race',
                ylabel='Subject Count',
                title='Subject Race by Gender')
        h, l = ax6.get_legend_handles_labels()

        gender_dict = {
            'Male':demographics_df.loc[demographics_df.sex=='Male'].shape[0],
            'Female':demographics_df.loc[demographics_df.sex=='Female'].shape[0]
        }

        new_labels = [f"{x} - {gender_dict.get(x)}" for x in l]
        
        ax6.legend(h,
                new_labels,
                title='Gender',
                loc='upper left',
                shadow=True)

        #figname = OUTPUT_DIR / f"{COLLECT_SET}_demographics_snapshot.png"
        #plt.savefig(figname)

        plt.show()


    def plotResolution(self):
        df = self.probe_df.copy()

        fig = plt.figure(constrained_layout=True, figsize=(10,10))
        #fig.suptitle(f'Demographics')

        plt.subplot(2,2,1)
        plt.hist(df.head_h_min,bins=np.arange(10,df.head_h_min.max()+10,10))
        plt.title("Head Height Distribution")
        plt.xlabel("Min Head Height (pixels)")
        plt.ylabel("Probe Media Count")

        plt.subplot(2,2,2)
        plt.hist(df.head_h_min,bins=np.arange(10,df.body_h_min.max()+10,10))
        plt.title("Body Height Distribution")
        plt.xlabel("Min Body Height (pixels)")
        plt.ylabel("Probe Media Count")

        plt.show()


    def summary(self,label=None):
        data = [
            [len(self.probe_df)],
            [len(self.gallery_df)],
            [self.mask_matrix.sum()],
            [(~self.mask_matrix).sum()],
            [len(self.score_matrix.flatten())],
        ]

        if label == None:
            label = self.label

        index = ['Probe Entries','Gallery Entries','Match Scores','Non-Match Scores','Total Scores']
        return pd.DataFrame(data,columns=[label],index=index)

    def __len__(self):
        '''
        @return: the size of the probe set
        '''
        return len(self.probe_df)


    def toSearchResults(self):
        '''
        Convert this data to search results.
        '''
        from SearchResults import SearchResults

        search_results = []
        for probe, row in zip(self.probe_df.iloc, self.score_matrix):
            results = list(zip(self.gallery_df['subject_id'],row))
            results.sort(key=lambda x: -x[1])
            ids,scores = zip(*results)
            ids = np.array(ids)
            scores = np.array(scores)
            search_results.append([probe['subject_id'],ids,scores])

        return SearchResults(search_results,self.probe_df,self.gallery_df,algorithm=self.algorithm,label=self.label,color=self.color)