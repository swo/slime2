#!/usr/bin/env python

'''
    slime2 -- synthetic learning in microbial ecology
    Copyright (C) 2015  Scott W. Olesen

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


    swo@mit.edu
'''

import argparse, cPickle as pickle, hashlib, sys, time, random, itertools
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

def hash_tag(strs, tag_length=6):
    '''create a short tag using md5'''

    m = hashlib.md5()
    m.update(''.join([s for s in strs]))
    return m.hexdigest()[0: tag_length]

def parse_table_and_classes(table, klasses_fn):
    '''
    read an OTU table and a file that specifies the classes for each sample.

    the table has qiime format (rows=otus, columns=samples).
    the classes file can take one of two formats: one-column or two-column.

    returns : (trimmed_table, classes)
        trimmed_table : pandas dataframe, with the samples without specified classes
            being dropped
        classes : list of classes in the same order as the indices in the dataframe
    '''

    # read in the classes files
    with open(klasses_fn) as f:
        lines = [l.rstrip() for l in f if not l.startswith("#")]

    # figure out how to parse the classes file
    n_fields = len(lines[0].split())
    if n_fields == 1:
        samples, klasses = parse_one_column_klasses(lines)
    elif n_fields == 2:
        samples, klasses = parse_two_column_klasses(lines)
    else:
        raise RuntimeError("got {} fields in classes file".format(n_fields))

    # read in the table
    raw_table = pd.read_table(table, index_col=0).transpose()

    # check to see that all the samples are columns
    # complain if the classes file gave a sample not that's not in the OTU table
    cols = list(raw_table.index.values)
    missing_cols = [s for s in samples if s not in cols]
    if len(missing_cols) > 0:
        raise RuntimeError("samples {} not a column in table, which has columns {}".format(missing_cols, cols))

    # only keep samples in the OTU table if they have classes associated with them
    trim_table = raw_table.loc[list(samples)]

    return trim_table, klasses

def parse_two_column_klasses(lines):
    '''
    two-column files have lines with two tab-separated fields: sample-tab-class.
    for example, lines would be like sick_guy1 tab sick, healthy_guy1 tab healthy, etc.
    '''

    samples, klasses = zip(*lines)
    return samples, klasses

def parse_one_column_klasses(lines, comment="#"):
    '''
    one-column files have a header line with the name of the class, then the
    samples in that class, then a blank line before the next class. for example,
    lines would be like: sick, sick_guy1, sick_guy2, blank, healthy, healthy_guy1, etc.
    '''

    samples = []
    klasses = []

    read_klass = True # flag for asking if the next non-comment line is a class name
    for line in lines:
        if line.startswith(comment):
            # ignore comment lines
            continue
        elif line == "":
            # the next line after a blank is a class name
            read_klass = True
        elif read_klass:
            # the next lines after the class are samples
            klass = line
            read_klass = False
        else:
            sample = line
            samples.append(sample)
            klasses.append(klass)

    return samples, klasses

def create_rfc(otu_table, klasses, sample_weights=None, **rfc_args):
    '''initialize rfc from otu table and class file'''

    # construct the random forest object and fit the data
    rfc = RFC(**rfc_args)
    rfc.fit(table, klasses, sample_weights)

    # attach some extra data to the random forest object for bookkeeping
    rfc.true_klasses = klasses
    rfc.predicted_klasses = rfc.predict(table)
    rfc.total_score = rfc.score(table, klasses)
    rfc.feature_names = list(table.columns.values)
    rfc.ordered_features = sorted(zip(rfc.feature_names, rfc.feature_importances_), key=lambda x: -x[1])

    return rfc

def tagged_name(fn, tag, suffix='txt'):
    '''format filenames for rfc output'''
    return "{}_{}.{}".format(tag, fn, suffix)

def categorize_classifications(targets, predictions):
    '''
    this is like an explicit confusion matrix. make a line for every sample.
    if a sample was X and was classified as X, just write "--". if it was X
    but classified as Y, write ">> X misclassified as Y".
    '''

    out = []
    for t, p in zip(targets, predictions):
        if t == p:
            out.append("--")
        else:
            out.append(">> {} misclassified as {}".format(t, p))

    return out

def save_results(rfc, tag):
    '''
    save the results of an rfc run. tag every output filename with a prefix
    so that they all appear next to each other in the directory.
    '''

    # pickle the whole rfc
    with open(tagged_name('rfc', tag, suffix='pkl'), 'w') as f:
        pickle.dump(rfc, f, protocol=2)

    # save the other information in text files
    with open(tagged_name('classes', tag), 'w') as f:
        f.write('\n'.join(rfc.predicted_klasses) + '\n')

    with open(tagged_name('featimp', tag), 'w') as f:
        cumul_imp = 0
        for of in rfc.ordered_features:
            cumul_imp += float(of[1])
            f.write("{}\t{}\t{:.3f}\n".format(of[0], of[1], cumul_imp))

    with open(tagged_name('scores', tag), 'w') as f:
        f.write("mean score: {}".format(rfc.total_score) + '\n')
        if hasattr(rfc, 'oob_score_'):
            f.write("oob score: {}".format(rfc.oob_score_) + '\n')

    with open(tagged_name('results', tag), 'w') as f:
        f.write('\n'.join(categorize_classifications(rfc.true_klasses, rfc.predicted_klasses)))

    with open(tagged_name('params', tag), 'w') as f:
        f.write('\n'.join(["{}: {}".format(*x) for x in rfc.get_params().items()]))

def int_or_none(x):
    '''
    take a string. if the string is 'none', return None object. if it's
    an integer, return that integer.
    '''

    assert(isinstance(x, str))
    if x.lower() == 'none':
        return None
    else:
        return int(x)

def int_float_str(x):
    '''
    take a string. if it's an integer, parse it that way. then try for a
    float. if that fails, just leave it as a string.
    '''

    assert(isinstance(x, str))
    try:
        return int(x)
    except ValueError, e:
        try:
            return float(x)
        except ValueError, e:
            return x

def assign_weights(weights_string, klasses):
    weight_vals = [float(x) for x in weights_string.split(",")]
    chunked_klasses = [g[0] for g in itertools.groupby(klasses)]
    weight_map = {k: w for w, k in zip(weight_vals, chunked_klasses)}

    weights = np.array([weight_map[k] for k in klasses])
    return weights


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="slime2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group('io')
    g.add_argument('otu_table')
    g.add_argument('classes', help='specifications of samples and their true classes')
    g.add_argument('--output_tag', '-o', default=None, help='tag for output data (default: use a hash tag)')
    g.add_argument('--rfc', '-c', default=None, help='use an existing classifier?')
    g.add_argument('--shuffle', action='store_true', help='shuffle class labels?')
    g.add_argument('--weights', help='set of floats, comma separated')

    g = p.add_argument_group('tree details')
    g.add_argument('--n_estimators', '-n', default=10, type=int, help='number of trees')
    g.add_argument('--criterion', default='gini', choices=['gini', 'entropy'], help='function to measure quality of split')
    g.add_argument('--max_features', '-f', type=int_float_str, default='auto')
    g.add_argument('--random_state', '-r', type=int_or_none, default='none', help='random seed (none=random)')
    g.add_argument('--max_depth', '-d', type=int_or_none, default='none', help='(none=no limit)')
    g.add_argument('--no_oob_score', '-b', dest='oob_score', action='store_false')
    g.add_argument('--n_jobs', '-j', type=int, default=1, help='-1=# of nodes')
    g.add_argument('--verbose', '-v', action='count', help="verbosity for the random forest creation and stdout summary")

    args = p.parse_args()

    tag = hash_tag([open(args.otu_table).read(), open(args.classes).read()])

    # save the command line
    with open(tagged_name('cmd', tag), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    table, klasses = parse_table_and_classes(args.otu_table, args.classes)

    if args.weights:
        sample_weights = assign_weights(args.weights, klasses)
    else:
        sample_weights = None

    if args.shuffle:
        random.shuffle(klasses)

    if args.rfc is None:
        # prepare the arguments for the random forest instantiation
        rfc_args = dict(vars(args))
        for a in ['otu_table', 'classes', 'output_tag', 'rfc', 'shuffle', 'weights']:
            del rfc_args[a]

        start_time = time.time()
        rfc = create_rfc(table, klasses, sample_weights, **rfc_args)

        if not args.shuffle:
            save_results(rfc, tag)
            print "saved results with tag {}".format(tag)

        end_time = time.time()

        if args.verbose > 0:
            print "walltime elapsed: {:.1f} seconds".format(end_time - start_time)

            if hasattr(rfc, 'oob_score_'):
                print "oob score: {:.5f}".format(rfc.oob_score_)

            print "top features:"
            for i in range(10):
                print "  {}\t{}".format(*rfc.ordered_features[i])
    else:
        with open(args.rfc) as f:
            rfc = pickle.load(f)

        # run this rfc with the new table and classes
        predicted_klasses = rfc.predict(table)

        for x in categorize_classifications(klasses, predicted_klasses):
            print x

        print "score: {}".format(rfc.score(table, klasses))