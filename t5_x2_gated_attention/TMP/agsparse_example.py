"""
--------------------------------------------------------------------------------
GENERAL PURPOSE TRAINER FOR NEGOTIATION MODELING PROJECT

    A framework to easily train/test with different strategies

--------------------------------------------------------------------------------
ARGUMENTS

    * --rm_architecture > [OPTIONAL, STRING]
        FORMAT
        DEFAULT: "T5ForConditionalGeneration"
    * --rm_tokenizer > [OPTIONAL, STRING]
        FORMAT
        DEFAULT: "T5Tokenizer"

    * --opponent_model > [OPTIONAL, STRING]
        FORMAT:

        SUB-ARGS:
            * --om_architecture > [OPTIONAL, STRING]
                FORMAT
                DEFAULT: "T5ForConditionalGeneration"
            * --om_tokenizer > [OPTIONAL, STRING]
                FORMAT
                DEFAULT: "T5Tokenizer"
            * --om_inclusion > [OPTIONAL, STRING]
                DEPENDENCIES: [response_model == True]
                FORMAT

    * --dataset > [OPTIONAL, STRING]
        FORMAT:

    * --output_dir > [OPTIONAL, STRING]
        FORMAT:

--------------------------------------------------------------------------------
"""

import argparse as ap


def arg_handler():
    # Choices for different arguments
    architecture_choices = ['T5ForConditionalGeneration']
    tokenizer_choices = ['T5Tokenizer']
    om_choices = ['preference']
    om_inclusion_choices = ['append_confidence']
    dataset_choices = ['casino_opponent_pref']
    output_dir_default = './output'

    parser = ap.ArgumentParser()

    # Response Model Options
    parser.add_argument('--rm_architecture', dest='rm_architecture',
                        type=str, required=False, default=architecture_choices[0], choices=architecture_choices)
    parser.add_argument('--rm_tokenizer', dest='rm_tokenizer',
                        type=str, required=False, default=tokenizer_choices[0], choices=tokenizer_choices)

    # Opponent Model Options
    parser.add_argument('--opponent_model', dest='opponent_model',
                        type=str, required=False, default=om_choices[0], choices=om_choices)

    parser.add_argument('--om_architecture', dest='om_architecture',
                        type=str, required=False, default=architecture_choices[0], choices=architecture_choices)
    parser.add_argument('--om_tokenizer', dest='om_tokenizer',
                        type=str, required=False, default=tokenizer_choices[0], choices=tokenizer_choices)
    parser.add_argument('--om_inclusion', dest='om_inclusion',
                        type=str, required=False, default=om_inclusion_choices[0], choices=om_inclusion_choices)

    # Dataset Options
    parser.add_argument('--dataset', dest='dataset',
                        type=str, required=False, default=dataset_choices[0], choices=dataset_choices)

    # Output Options
    parser.add_argument('--output_dir', dest='output_dir',
                        type=str, required=False, default=output_dir_default)

    return parser.parse_args()


def dispatcher(args: ap.Namespace):
    pass


def main():
    '''
    TODO
        Look into people (papers) using gates with pretrained models
            try to do something with leaky cross attention
            Share with kushal when find
        expolore adaptors thing 
    '''
    args = arg_handler()
    dispatcher(args)


if __name__ == '__main__':
    main()
