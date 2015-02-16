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

import argparse, cPickle as pickle, hashlib, sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC

def hash_tag(fns, tag_length=6):
    '''create a short tag using md5'''

    m = hashlib.md5()
    m.update(''.join([open(fn).read() for fn in fns]))
    return m.hexdigest()[0: tag_length]

def create_rfc(otu_table, klasses_fn, **rfc_args):
    table = pd.io.parsers.read_table(otu_table)
    klasses = [line.rstrip() for line in open(klasses_fn).readlines()]

    rfc = RFC(**rfc_args)
    rfc.fit(table, klasses)
    rfc.true_klasses = klasses
    rfc.predicted_klasses = rfc.predict(table)
    rfc.total_score = rfc.score(table, klasses)

    return rfc

def tagged_name(fn, tag, suffix='txt'):
    return "{}_{}.{}".format(tag, fn, suffix)

def categorize_classifications(targets, predictions):
    out = []
    for t, p in zip(targets, predictions):
        if t == p:
            out.append("--")
        else:
            out.append(">> {} misclassified as {}".format(t, p))

    return out

def save_results(rfc, tag):
    # pickle the whole rfc
    with open(tagged_name('rfc', tag, suffix='pkl'), 'w') as f:
        pickle.dump(rfc, f)

    # save the other information in text files
    with open(tagged_name('classes', tag), 'w') as f:
        f.write('\n'.join(rfc.predicted_klasses) + '\n')

    with open(tagged_name('featimp', tag), 'w') as f:
        f.write('\n'.join([str(x) for x in rfc.feature_importances_]) + '\n')

    with open(tagged_name('scores', tag), 'w') as f:
        f.write("mean score: {}".format(rfc.total_score) + '\n')
        if hasattr(rfc, 'oob_score_'):
            f.write("oob score: {}".format(rfc.oob_score_) + '\n')

    with open(tagged_name('results', tag), 'w') as f:
        f.write('\n'.join(categorize_classifications(rfc.true_klasses, rfc.predicted_klasses)))

    with open(tagged_name('params', tag), 'w') as f:
        f.write('\n'.join(["{}: {}".format(*x) for x in rfc.get_params().items()]))

def int_or_none(x):
    assert(isinstance(x, str))
    if x.lower() == 'none':
        return None
    else:
        return int(x)

def int_float_str(x):
    assert(isinstance(x, str))
    try:
        return int(x)
    except ValueError, e:
        try:
            return float(x)
        except ValueError, e:
            return x


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="slime2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group('io')
    g.add_argument('otu_table')
    g.add_argument('classes', help='newline-separated list of target classes')
    g.add_argument('--output_tag', '-o', default=None)

    g = p.add_argument_group('tree details')
    g.add_argument('--n_estimators', '-n', default=10, type=int, help='number of trees')
    g.add_argument('--criterion', default='gini', choices=['gini', 'entropy'], help='function to measure quality of split')
    g.add_argument('--max_features', '-f', type=int_float_str, default='auto')
    g.add_argument('--random_state', '-r', type=int_or_none, default='none', help='random seed (none=random)')
    g.add_argument('--max_depth', '-d', type=int_or_none, default='none', help='(none=no limit)')
    g.add_argument('--no_oob_score', '-b', dest='oob_score', action='store_false')
    g.add_argument('--n_jobs', '-j', type=int, default=1, help='-1=# of nodes')

    args = p.parse_args()

    tag = hash_tag([args.otu_table, args.classes])

    # save the command line
    with open(tagged_name('cmd', tag), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    # prepare the arguments for the random forest instantiation
    rfc_args = dict(vars(args))
    for a in ['otu_table', 'classes', 'output_tag']:
        del rfc_args[a]

    rfc = create_rfc(args.otu_table, args.classes, **rfc_args)
    save_results(rfc, tag)
    print "saved results with tag {}".format(tag)
