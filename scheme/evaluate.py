#!/usr/bin/env python3

import csv
import os
import re
import sys
from itertools import groupby

import configargparse
from ucca import evaluation, ioutil

from scheme.cfgutil import add_verbose_argument
from scheme.convert import CONVERTERS, UCCA_EXT
from scheme.evaluation import amr, sdp, conllu

desc = """Parses files in any format, and evaluates using the proper evaluator."""


EVALUATORS = {
    None: evaluation,
    "sdp": sdp,
    "conllu": conllu,
    "amr": amr,
}


class Scores(object):
    """
    Keeps score objects from multiple formats
    """
    def __init__(self, scores):
        self.scores_by_format = [(t, t.aggregate(s)) for t, s in groupby(scores, type)]

    @staticmethod
    def name():
        return "Multiple formats"

    @staticmethod
    def aggregate(scores):
        return Scores([s for score in scores for _, s in score.scores_by_format])

    def average_f1(self, *args, **kwargs):
        return sum(s.average_f1(*args, **kwargs) for t, s in self.scores_by_format) / len(self.scores_by_format)

    def print(self, *args, **kwargs):
        for t, s in self.scores_by_format:
            if len(self.scores_by_format):
                print(name(t) + ":", *args, **kwargs)
            s.print(*args, **kwargs)

    def fields(self):
        return [f for _, s in self.scores_by_format for f in s.fields()]

    def titles(self):
        return [(name(t) + "_" + f) for t, s in self.scores_by_format for f in s.titles()]

    def __str__(self):
        print(",".join(self.fields()))


def name(t):
    return t.name() if hasattr(t, "name") else "UCCA"


def passage_format(filename):
    basename, ext = os.path.splitext(os.path.basename(filename))
    return basename, None if ext in UCCA_EXT else ext.lstrip(".")


def read_files(files, default_format=None):
    for filename in sorted(files, key=lambda x: tuple(map(int, re.findall("\d+", x))) or x):
        basename, converted_format = passage_format(filename)
        in_converter, out_converter = CONVERTERS.get(converted_format, CONVERTERS[default_format])
        if in_converter:
            with open(filename, encoding="utf-8") as f:
                for _, passage, passage_id in in_converter(f, passage_id=basename, return_original=True):
                    yield passage, passage_id, converted_format, in_converter, out_converter
        else:
            passage = ioutil.file2passage(filename)
            yield passage, passage.ID, converted_format, in_converter, out_converter


def evaluate_all(args, evaluate, files):
    for ((guessed_passage, _, guessed_format, guessed_converter, _),
         (ref_passage, passage_id, ref_format, _, ref_converter)) in zip(*[read_files(f, args.format) for f in files]):
        if not args.quiet:
            print(passage_id, end=" ")
        if guessed_format != ref_format:
            guessed_passage = next(iter(guessed_converter(guessed_passage + [""], passage_id=passage_id))) if \
                ref_converter is None else ref_converter(guessed_passage)
        result = evaluate(guessed_passage, ref_passage, verbose=args.verbose > 1)
        if not args.quiet:
            print("F1: %.3f" % result.average_f1())
        if args.verbose:
            result.print()
        yield result


def write_csv(filename, rows):
    if filename:
        with sys.stdout if filename == "-" else open(filename, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)


def main(args):
    files = [[os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else [d] for d in (args.guessed, args.ref)]
    evaluate = EVALUATORS.get(passage_format(files[1][0])[1], EVALUATORS[args.format]).evaluate
    results = list(evaluate_all(args, evaluate, files))
    summary = Scores(results)
    if len(results) > 1:
        if args.verbose:
            print("Aggregated scores:")
        if not args.quiet:
            print("F1: %.3f" % summary.average_f1())
            summary.print()
    elif not args.verbose:
        summary.print()
    write_csv(args.out_file,     [summary.titles()] + [result.fields() for result in results])
    write_csv(args.summary_file, [summary.titles(), summary.fields()])


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", help="filename/directory for the guessed annotation(s)")
    argparser.add_argument("ref", help="filename/directory for the reference annotation(s)")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    argparser.add_argument("-o", "--out-file", help="file to write results for each evaluated passage to, in CSV format")
    argparser.add_argument("-s", "--summary-file", help="file to write aggregated results to, in CSV format")
    group = argparser.add_mutually_exclusive_group()
    add_verbose_argument(group, help="detailed evaluation output")
    group.add_argument("-q", "--quiet", action="store_true", help="do not print anything")
    main(argparser.parse_args())
