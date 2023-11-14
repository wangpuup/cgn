# -*- coding: utf-8 -*-
"""
Created on Jan 13, 2021

@author: compi

For options, run command
>> python score.py -h

"""
import os, sys, re
import numpy as np
import pandas as pd
import argparse

def read_corpus(fname):
    """
    reads a corpus file 
    each line consists of an utterance with optionally preceding utterance id's
    blanco lines are skipped
    For files without IDs numeric IDs of the form #nnn are returned

    Parameters
    ----------
        fname : str
            file name
        ID_tag : boolean, default=True
            if True utterance ID's are the first word in a sentence

    Returns
    -------
        ids : list of str
            list of ids
        utt : list of str
            list of utterances
    """
    #fp = open(fname,"r",encoding="utf-8")
    #lines = fp.read().splitlines()
 
    text = []
    ids = []
    with open(fname,"r") as f:
        for ln in f.readlines():
            k = ln.split("(",1)[1]
            v = ln.split("(",1)[0]
            ids.append( k)
            text.append(v)
    # normalize the id's to lowercase
    return([id.lower() for id in ids],text)

# tokenizer converts a text to tokens
def tokenizer(text):
    tokens = text.strip().split()
    return(tokens)

def levenshtein(seq1, seq2):
    '''
    Levenshtein Distance
        Finds the symmetric Levenshtein distance as the sum of INS/DEL/SUB
        There is no backtracking, and no seperate maintenance of INS/DEL/SUB
        (reduced CPU load > 30% compared to levenshtein_v01)

    Parameters
    ----------
        seq1 : list
            tokens in list1 (either hypothesis or test)
        seq2 : list
            tokens in list2 (the other)

    Returns
   --------
        cummdist : int
            total number of edits
    '''
    Nx = len(seq1) 
    Ny = len(seq2) 
    prev = np.zeros(Ny+1)
    current = np.zeros(Ny+1)
    
    for j in range(0,Ny+1):
        current[j] = j

    for i in range(1, Nx+1):
        prev = current.copy()
        current[0] = prev[0]+1
        for j in range(1, Ny+1):
            if seq1[i-1] == seq2[j-1]:
                current[j] = min( prev[j]+1, prev[j-1], current[j-1]+1)
            else:    
                current[j] = min( prev[j], prev[j-1], current[j-1] ) + 1      
    return (current[Ny])


def levenshtein_v01(seq1, seq2):
    '''
    Levenshtein Distance
        Finds the symmetric Levenshtein distance as the sum of INS/DEL/SUB

    Parameters
    ----------
        seq1 : list
            tokens in list1 (either hypothesis or test)
        seq2 : list
            tokens in list2 (the other)
    
    Returns
   --------
        cummdist : int
            total number of edits
    '''
    Nx = len(seq1) 
    Ny = len(seq2) 
    trellis= np.zeros ((Nx+1, Ny+1))
    for i in range(0,Nx+1):
        trellis[i, 0] = i
    for j in range(0,Ny+1):
        trellis[0, j] = j

    for i in range(1, Nx+1):
        ii=i-1
        for j in range(1, Ny+1):
            jj = j-1
            if seq1[ii] == seq2[jj]:
                trellis[i,j] = min(
                    trellis[i-1, j] + 1,
                    trellis[i-1, j-1],
                    trellis[i, j-1] + 1
                )
            else:
                trellis[i,j] = min(
                    trellis[i-1,j] + 1,
                    trellis[i-1,j-1] + 1,
                    trellis[i,j-1] + 1
                )
    return (trellis[Nx , Ny ])


def dtw(x=[],y=[],norm=None,wS=4.,wI=3.,wD=3.,wC=1., Compounds=False, Dash=False, Verbose=False):
    '''
    Weighted Edit Distance by DTW aligment allowing for SUB/INS/DEL

    Parameters
    ----------
    x : list (or str) 
        tokens in test
    y : list (or str)
        tokens in reference
    norm : Normalizer object, default=None
        A text normalization object from the Normalizer class

    wS, wI, wD, wC : float, default (4., 3., 3., 2.)
        edit costs for Substition, Insertion, Deletion and Compound
    
    Compound : boolean, default=False
        if True Compounding is allowed

    Dash : boolean, default=False
        if True (and Compound is True) dash-compounding is allowed

    Verbose : boolean, default=False
        if True highly Verbose printing of internal results (trellis, backtrace, .. )


    Returns
    -------
        cummdist : float
            weighted edit distance
        alignment : DataFrame
            alignment path as DataFrame with columns [ 'x', 'y', 'OPS' ]
        details : list of int
            counts of [nsub,nins,ndel,Ny,nCx,nCy]
            if Compounds is False then last 2 arguments will be 0
             
    '''
    Nx = len(x) 
    Ny = len(y) 
    trellis = np.zeros((Nx+1, Ny+1))
    bptr = np.zeros((Nx+1, Ny+1, 2), dtype='int')
    edits = np.full((Nx+1, Ny+1),"Q",dtype='str')
     
    for i in range(1,Nx+1):
        trellis[i, 0] = i * wI
        bptr[i,0] = [i-1,0]
        edits[i,0] = 'I'
    for j in range(1,Ny+1):
        trellis[0, j] = j * wD
        bptr[0,j] = [0,j-1]
        edits[0,j] = 'D'

    # forward pass - trellis computation
    # indices i,j apply to the trellis and run from 1 .. N
    # indices ii,jj apply to the data sequence and run from 0 .. N-1
    for i in range(1, Nx+1):
        ii=i-1
        for j in range(1, Ny+1):
            jj=j-1

            # substitution or match
            score_SUB = trellis[i-1,j-1] + int(x[ii]!=y[jj]) * wS
            trellis[i,j] = score_SUB
            bptr[i,j] = [i-1,j-1]         
            if (x[ii]==y[jj]): edits[i,j] = 'M'            
            else: edits[i,j] = 'S'
                
            # insertion and deletions
            score_INS = trellis[i-1,j] + wI            
            if( score_INS < trellis[i,j] ):
                trellis[i,j] = score_INS
                bptr[i,j] = [i-1,j]
                edits[i,j] = 'I'
            score_DEL = trellis[i,j-1] + wD                
            if( score_DEL < trellis[i,j] ):
                trellis[i,j] = score_DEL
                bptr[i,j] = [i,j-1]
                edits[i,j] = 'D'
    
            # compounds
            if( Compounds & (ii>0) ):
                Cx = ( (x[ii-1]+x[ii]) == y[jj] )
                if( Dash ):
                    Cx |= ( (x[ii-1]+'-'+x[ii]) == y[jj] )
                score_Cx = trellis[i-2,j-1] + wC
                if( Cx & (score_Cx < trellis[i,j]) ):
                    trellis[i,j] = score_Cx
                    bptr[i,j] = [i-1,j-1]
                    edits[i,j] = 'C'
                    bptr[i-1,j-1] = [i-2,j-1]
                    edits[i-1,j-1] = 'X'
            if( Compounds & (jj>0) ):
                Cy = ( x[ii] == (y[jj-1]+y[jj]) )
                if( Dash ):
                    Cy |= ( x[ii] == (y[jj-1]+'-'+y[jj]) )                
                score_Cy = trellis[i-1,j-2] + wC
                if( Cy & (score_Cy < trellis[i,j]) ):
                    trellis[i,j] = score_Cy
                    bptr[i,j] = [i-1,j-1]
                    edits[i,j] = 'C'
                    bptr[i-1,j-1] = [i-1,j-2]
                    edits[i-1,j-1] = 'Y'
            
                    
    # backtracking
    (ix,iy) = bptr[Nx,Ny]
    trace = [ (Nx-1,Ny-1) ]
    while( (ix>0) | (iy>0) ):
        trace.append( (ix-1,iy-1) )
        (ix,iy) = bptr[ix,iy]   
    trace.reverse()
    
    # recovering alignments as [ ( x_i, y_j, edit_ij ) ]
    # the dummy symbol '_' is inserted for counterparts of insertions, deletions
    #     and first element in compounds
    alignment = []
    for k in range(len(trace)):
        (ix,iy) = trace[k]
        ops = edits[ix+1,iy+1]
        if (ops == 'I') | (ops == 'X'): 
            alignment.append( [ x[ix] , '_' , ops ] )
        elif (ops == 'D') | (ops == 'Y'): 
            alignment.append( [ '_' , y[iy] , ops ] ) 
        elif (ops == 'M') | (ops == 'S') | (ops == 'C'): 
            alignment.append( [ x[ix] , y[iy] , ops ] )
            
    nins = sum([ int(alignment[i][2]=='I') for i in range(len(alignment))])
    ndel = sum([ int(alignment[i][2]=='D') for i in range(len(alignment))])
    nsub = sum([ int(alignment[i][2]=='S') for i in range(len(alignment))])
    nxcomp = sum([ int(alignment[i][2]=='X') for i in range(len(alignment))])        
    nycomp = sum([ int(alignment[i][2]=='Y') for i in range(len(alignment))])
    alignment_df = pd.DataFrame(alignment,columns=['x','y','O'])
    
    if (Verbose):
        print("Edit Distance: ", trellis[Nx,Ny])
        print(trellis[0:,0:].T)
        print(edits[0:,0:].T)
        print(trace)
        print(alignment_df.transpose())
        print("Number of Words: ",Ny)
        print("Substitutions/Insertions/Deletions: ",nsub,nins,ndel)
        print("X-Compounds/Y-Compounds: ", nxcomp,nycomp )

    return(trellis[Nx , Ny],alignment_df,[nsub,nins,ndel,Ny,nxcomp,nycomp])


def score_corpus(hyp_fname,ref_fname,CER=False,ignore_list=None,norm=None,
               Compounds=False,Dash=False,Print_Align=False,Print_Errors=False,Logdir=None):
    ''' 
    score the output of a speech recognition experiment
    the reference and hypothesis transcriptions are paragraphs
    the assumption is that lines are separated by LF and each utterance/sentence is preceded by an utterance-ID
    the scoring will be done for all utterances in the hypothesis file,
        the reference file may contain more utterances than the hypothesis file
        utterance ID's found in the ignore list will be omitted    

    Parameters
    ----------
        hyp_fname : str
            Hypothesis File Name
        ref_fname : str
            Reference File Name
        CER : boolean, default=False
            Compute Character Error Rate instead of Word Error Rate
        ignore_list : str
            File Name with ignore list
        norm : Normalization object
            Text Normalization
        Compounds: boolean, default=False
            if True Compounding is allowed
        Dash: boolean, default=False
            if True dash-compounding is allowed
        Print_Align : boolean, default=False
            if True an alignment is printed for each file
        Print_Errors : boolean, default=False
            if True print the error types for each file
        Logdir : str, default=None
            Path for output files, if not None print alignment and errors to files

    Returns
    -------
        error_rate : float
            Error Rate
        ntot_words : int
            Number of words in the reference
        all_errors : pd.DataFrame
            DataFrame containing all errors and close errors (compounds) 
    '''
    if CER:
        if Compounds:
            print(" WARNING: COMPOUNDING ignored for Character Error Rate Computation ")
            Compounds = False
        if Dash:
            print(" WARNING: DASH COMPOUNDING ignored for Character Error Rate Computation ")
            Dash = False

    if Logdir is not None:
        os.makedirs(Logdir, exist_ok=True)
        align_f = open(os.path.join(Logdir, 'alignments'), 'w')
        errors_f = open(os.path.join(Logdir, 'errors'), 'w')
        results_f = open(os.path.join(Logdir, 'results'), 'w')
        error_dict_f = open(os.path.join(Logdir, 'error_per_id'), 'w')
    
    # 1. Read reference and hypothesis texts 
    (hyp_ids,hyp_utt_raw) = read_corpus(hyp_fname)
    (ref_ids,ref_utt_raw) = read_corpus(ref_fname)

    try:
        (ignore_ids, ignore_utt_raw) = read_corpus(ignore_list)
    except:
        ignore_ids = []
    
    # 2. Normalization ...
    
    if norm is not None:
        hyp_utt = [norm.norm(utt) for utt in hyp_utt_raw]
        ref_utt = [norm.norm(utt) for utt in ref_utt_raw]
    else:
        hyp_utt = hyp_utt_raw
        ref_utt = ref_utt_raw

    # 3. Matching
    reference = dict(zip(ref_ids,ref_utt))
    ntot_words = 0
    ntot_sub = 0
    ntot_ins = 0
    ntot_del = 0
    ntot_Cx = 0
    ntot_Cy = 0
    ntot_utt = 0
    all_errors = pd.DataFrame(columns=['x','y','O'])

    error_dict = {}
    for i in range(len(hyp_ids)):
        id = hyp_ids[i]
        if id in ignore_ids: continue
        #if(Verbose): print(id,hyp_utt[i],reference[id])
        x = hyp_utt[i] if CER else tokenizer(hyp_utt[i])
        try:
            y = reference[id] if CER else tokenizer(reference[id])
            _,align,(nsub,nins,ndel,Ny,nCx,nCy) =  \
                    dtw(x,y,Compounds=Compounds,Dash=Dash,Verbose=False)
            error_dict[id] = (Ny, nins, nsub, ndel)
            utt_errors = align.loc[align['O']!='M']
            if(Print_Align): print(id,'\n',align.transpose())
            if(Print_Errors): print(utt_errors.transpose()) 
            if Logdir is not None:
                print(id,'\n',align.transpose(),file=align_f)
                print(utt_errors.transpose(),file=errors_f)
            all_errors = pd.concat([all_errors,utt_errors])
            ntot_words += Ny
            ntot_del += ndel
            ntot_sub += nsub
            ntot_ins += nins
            ntot_Cx += nCx
            ntot_Cy += nCy
            ntot_utt += 1
        except KeyError:
            pass
            #print("WARNING:  Hypothesis with id=%s has no matching Reference" % hyp_ids[i])

    print("===== SUMMARY =================================")
    if CER: 
        print("--- Reporting Character Error Rates ---")
    else: 
        print("--- Reporting Word Error Rates ---")
    ntot_err1 = ntot_sub+ntot_ins+ntot_del
    ntot_err2 = ntot_Cx + ntot_Cy
    ntot_words_adjusted = ntot_words - ntot_Cy
    err1_rate = 100.0*float(ntot_err1)/float(ntot_words)
    err2_rate = 100.0*float(ntot_err2)/float(ntot_words)
    print("#Sentences: %d" % ntot_utt)
    print("#Words: %d " % (ntot_words) )
    print("Error_Rate: %.2f%% S+I+D  + %.2f%% C"   % (err1_rate,err2_rate) )
    print("Details: (#S #I #D #C) %d %d %d %d+%d" % (ntot_sub,ntot_ins,ntot_del,ntot_Cx,ntot_Cy)) 
    print("===============================================")   
    
    if Logdir is not None:
        print("#Sentences: %d" % ntot_utt, file=results_f)
        print("#Words: %d " % (ntot_words), file=results_f)
        print("Error_Rate: %.2f%% S+I+D  + %.2f%% C"   % (err1_rate,err2_rate), file=results_f)
        print("Details: (#S #I #D #C) %d %d %d %d+%d" % (ntot_sub,ntot_ins,ntot_del,ntot_Cx,ntot_Cy), file=results_f)
        print('ID #words #ins #subs #del', file=error_dict_f)
        id_list = list(dict(sorted(error_dict.items(), key=lambda item: item[1][0])).keys())
        for id in id_list:
            N, I, S, D = error_dict[id]
            print('%s %i %i %i %i' % (id,N,I,S,D), file=error_dict_f) 
        error_dict_f.close()
        errors_f.close()
        align_f.close()
        results_f.close()

    

    return(err1_rate,ntot_words,all_errors)

class Normalizer():
    """
    Summary
    -------
        The Normalizer class delivers a set of text normalization operations.

        The Normalizer functions as follows:
        1. First all required resources need to be preloaded to the class object
        2. The Normalizer.pipe process pipeline needs to be defined
        3. The Normalizer.norm() method  can be run on any string

        Example:  you want to (a) reduce white space to single characters (b) do number rewriting in Dutch
            (c) do some spelling normalization and (d) normalize the fillers to just a few symbols

            > Norm = Normalizer
            > Norm.pattern_subs = ['getallen100.lst']
            > Norm.word_subs = ['spelling.lst', 'fillers.lst']
            > Norm.load()
            > Norm.pipe = ['single_space','sub_words','sub_patterns']
            > normalized_string = Norm.norm(raw_string)

    Attributes
    ----------
        pipe : list,
            list with processing steps,
            default = ['single_space','sub_patterns','sub_words', 'strip_hyphen', 'del_tag', 'lower']

        word_subs : list,
            resources for word substitution,
            default = ['resources/nl_fillers.lst', 'resources/nl_abbrev.lst', 'resources/nl_nbest.lst']

        pattern_subs : list,
            resources for pattern substitution,
            default = ['resources/nl_getallen100.lst']

        _word_subs :  dict
            dictionary with active word substitutions
        _pattern_subs :  list,
            list of tuples with active pattern substitutions

    Named elementary pipeline operations
    ------------------------------------
        lower           convert to lower case
        upper           convert to upper case
        single_space    converts all whitespace and CR/LF segments to single space
        sub_words       apply word substitution from word_subs dictionary (no wildcards)
        sub_patterns    sequentially apply pattern substitution from pattern_subs list (no wildcards)
        del_tags        removes all tags, i.e. constructs of the form <???> with a max of 32 chars long
        strip_hyphen    removes hyphens attached to words (front or back) 
        split_hyphen    splits words on hyphens
        decomp_hyphen   replaces hyphens (word initial, internal or back)
                        by appropriate '_' compound indicators and split if word internal
        force_compounds concatenate xxx_ yyy | xxx _yyy | xxx_ _yyy to xxxyyy

    Special characters
    -------------------
        '_'     : word initial or word final underscores are compound indicators
        '-'     : is defined in the hyphen related operations
        '|'     : is the split character in the substitution files
        '<>'    : enclosing brackets for tags,

        _,|,<,> : should definitely not be used in the regular word character set

    Additional Notes on pattern vs. word substitutions
    --------------------------------------------------
        - pattern substitutions are ORDERED.
            They are applied sequentially for each pattern in the list
            All matched patterns are substitued. All characters can be part of a matching or substitution string.
            Hence the pattern matching applies to words, word internal and multiword fragments
        - word substitutions are UNORDERDED.
            They are applied sequentially for each word in the input string.
            White space characters are excluded.
            Each word in the input can be modified at most once.

    """

    def __init__(self,pipe=None,word_subs=None,pattern_subs=None,resourcesdir=None):
        
        # set the defaults
        if pipe is None:
            pipe = ['single_space', 'sub_patterns', 'sub_words', 'strip_hyphen', 'del_tags', 'lower']
        if resourcesdir is None:
            resourcesdir = './resources'
        if word_subs is None:
            word_subs = ['nl_fillers.lst', 'nl_abbrev.lst', 'nl_nbest.lst']
        if pattern_subs is None:
            pattern_subs = ['nl_getallen100.lst']

        word_subs = [os.path.join(resourcesdir, subf) for subf in word_subs]
        pattern_subs = [os.path.join(resourcesdir, subf) for subf in pattern_subs]

        self.pipe = pipe
        self.word_subs = word_subs
        self.pattern_subs = pattern_subs
        self._word_subs = {}
        self._pattern_subs = []
        self.load()

    def info(self):
        print("-- Normalizer Pipeline --")
        print(self.pipe)
        print("-- Loaded Normalizer Word Substitions --")
        print(self._word_subs)
        print("-- Loaded Normalizer Pattern Substitutions --")
        print(self._pattern_subs)

    def load(self):
        """
        load all named resources and add them to the active lists in word_subs and pattern_subs files
        """

        for fname in self.word_subs:
            self.load_word_subs(fname)
        for fname in self.pattern_subs:
            self.load_pattern_subs(fname)

    def load_word_subs(self, filename):
        """
        add the word substitutions in filename to the active list
        fileformat: lines with
            pattern|replacement|
        """

        with open(filename, encoding="utf-8") as f:
            for line in f:
                try:
                    (p, s) = line.rstrip().split("|", 1)
                    self._word_subs[p.strip()] = s.rstrip("|").strip()
                except ValueError:
                    print("WARNING(load_word_subs()): Skipping blanco line in : ", filename)

    def load_pattern_subs(self, filename):
        """
        add the pattern substitutions in filename to the active list
        fileformat: lines with
            pattern|replacement|
        """

        with open(filename, encoding="utf-8") as f:
            for line in f:
                try:
                    (p, s) = line.rstrip().split("|", 1)
                    self._pattern_subs.append((p, s.rstrip("|")))
                except ValueError:
                    print("Skipping line:", line)

    def norm(self, input_str, pipe=None):
        """
        Apply text normalization to the input

        Parameters
        ----------
            input_str : str
                input string to be normalized
Soc_S3_A_1.0.eaf.ManueleSegmentatieASRAnnotated1.txt
            pipe : list, optional
                pipeline to be processed, overriding the internally stored pipe

        Returns
        -------
            output_str : str
                normalized string
        """

        if pipe is None:
            pipe = self.pipe
        output_str = input_str

        for proc in pipe:

            # print(">>processing:",proc)
            if proc == "lower":
                output_str = output_str.lower()

            elif proc == "upper":
                output_str = output_str.upper()

            elif proc == "sub_patterns":
                for (p, s) in self._pattern_subs:
                    output_str = output_str.replace(p, s)

            elif proc == "sub_words":
                re_split1 = re.compile(r'(\S+)')
                items = re_split1.split(output_str)
                # replace item if found in the dictionary, otherwise use itself as default mapping
                output_str = "".join([self._word_subs.get(it, it) for it in items])

            elif proc == "del_tags":
                output_str = re.sub(r'<[^>]{0,32}>', r'', output_str)

            elif proc == "strip_hyphen":
                output_str = re.sub(r' -(\S)', r' \1', re.sub(r'(\S)- ', r'\1 ', output_str))

            elif proc == "split_hyphen":
                output_str = re.sub(r'(\S)-(\S)', r'\1 \2',output_str)

            elif proc == "decomp_hyphen":
                output_str = re.sub(r'-(\S)', r'_\1', re.sub(r'(\S)-', r'\1_', re.sub(r'(\S)-(\S)', r'\1_\2', output_str)))

            elif proc == "force_compounds":
                output_str = output_str.replace("_  _", "").replace("_ ", "").replace(" _", "")

            elif proc == "single_space":
                output_str = (" ".join(output_str.split()))

            else:
                print('WARNING: norm(): pipe-element %s not recognized', proc)

        return (output_str)

    def apply_norm(self, input, output):
        result = []
        for id, utt in list(zip(input[0], input[1])):
            result.append(id+' '+self.norm(utt)+'\n')

        with open(output, 'w') as pd:
            pd.writelines(result) 


def parseArgs():
    parser = argparse.ArgumentParser(description='Advanced scoring script for evaluation of speec recognition systems')
    group_sc = parser.add_argument_group('Scoring')
    group_sc.add_argument('-i', '--trans', required=True, help='Input transcription file')
    group_sc.add_argument('-t', '--ref', required=True, help='Reference transcription')
    group_sc.add_argument('-m', '--mode', default='wer', choices=['wer','cer'], help='Compute wer (word) or cer (character) error rate')
    group_sc.add_argument('-g', '--ignore', help='List with utterances to ignore')
    group_sc.add_argument('-c', '--compounds', action='store_true', help='If compounding is allowed')
    group_sc.add_argument('-d', '--dash', action='store_true', help='If dash-compounding is allowed')
    group_sc.add_argument('-a', '--align', action='store_true', help='Print the generated alignments')
    group_sc.add_argument('-e', '--errors', action='store_true', help='Print all errors')
    group_sc.add_argument('-n', '--norm', action='store_true', help='Apply normalisation')
    group_sc.add_argument('-o', '--outdir', help='Path if you wish to save alignment and errors in files, not by default')

    group_no = parser.add_argument_group('Normalisation')
    group_no.add_argument('-l', '--pipe', nargs='*', help='Pipeline normalisation steps you wish to apply (overriding the default, separate by space)')
    group_no.add_argument('-w', '--word_subs', nargs='*', help='Word substition lists to apply (overriding the default, separate by space)')
    group_no.add_argument('-p', '--pattern_subs', nargs='*', help='Pattern substitutions lists to apply (overriding the default, separate by space)')
    group_no.add_argument('-r', '--resourcesdir', help='Directory containg substitution lists')  
    group_no.add_argument('--output-wer-only', default=False, action='store_true', help='Only output the WER score, nothing else')

    args = parser.parse_args()

    assert os.path.exists(args.trans), 'Input transcription file does not exist!'
    assert os.path.exists(args.ref), 'Reference transcription file does not exist!'

    assert str(args.mode).lower() in ['cer', 'wer'], '-m/--mode should be cer or wer'

    return args

def main():
    args = parseArgs()
    print(vars(args),'\n')
    
    if args.norm:
        Norm = Normalizer(
            pipe=args.pipe,
            word_subs=args.word_subs,
            pattern_subs=args.pattern_subs,
            resourcesdir = args.resourcesdir)
        #Norm.info()
    else:
        Norm = None

    if str(args.mode).lower() == 'cer':
        CER = True
    else:
        CER = False

    err, N, df_err = score_corpus(args.trans, args.ref, CER=CER, ignore_list=args.ignore, norm=Norm, Compounds=args.compounds, Dash=args.dash, Print_Align=args.align, Print_Errors=args.errors, Logdir=args.outdir)

    return err

if __name__ == '__main__':
    main()
