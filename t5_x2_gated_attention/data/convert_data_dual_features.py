import json
import re
import os
import random


DESTINATION_DIR = 'splits/casino_dual_features'

SPLIT_TRAIN = .8
SPLIT_EVAL = .1
SPLIT_TEST = .1

ACCEPT_DEAL_SENTENCE = 'this deal is acceptable.'
WALKAWAY_SENTENCE = 'i am walking away from this deal'


def fix_train_test_split():
    global SPLIT_TRAIN, SPLIT_EVAL, SPLIT_TEST
    sum = SPLIT_TRAIN + SPLIT_EVAL + SPLIT_TEST
    SPLIT_TRAIN /= sum
    SPLIT_EVAL /= sum
    SPLIT_TEST /= sum


def get_submit_deal_sentence(task_data):
    task_dat_you = task_data['issue2youget']
    task_dat_them = task_data['issue2theyget']
    return f"lets submit this deal. i get {task_dat_you['Food']} food {task_dat_you['Water']} water and {task_dat_you['Firewood']} firewood. you get {task_dat_them['Food']} food {task_dat_them['Water']} water and {task_dat_them['Firewood']} firewood."


def remove_bad_chars(input: str):
    out = re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9\.\!\? ]+', '', input)).strip()
    return out if len(out) > 0 else " "


def get_context(agent_info: dict):
    '''
    # PERSONALITY STUFF
    agent_personality = agent_info['personality']
    b5_dict = agent_personality['big-five']
    big_five = ''
    for s in sorted(b5_dict, key=b5_dict.get, reverse=True):
        if s == 'emotional-stability':
            big_five += ' ' + 'stability'
        elif s == 'openness-to-experiences':
            big_five += ' ' + 'openess'
        else:
            big_five += ' ' + s
    '''

    agent_pref = agent_info["value2issue"]
    agent_reasons = agent_info['value2reason']
    hp_sentence = f'my highest priority is {agent_pref["High"]} because {remove_bad_chars(agent_reasons["High"])}'
    mp_sentence = f'my medium priority is {agent_pref["Medium"]} because {remove_bad_chars(agent_reasons["Medium"])}'
    lp_sentence = f'my lowest priority is {agent_pref["Low"]} because {remove_bad_chars(agent_reasons["Low"])}'
    
    return f"<CONTEXT> {hp_sentence} {mp_sentence} {lp_sentence}"


def get_opponent_context_sentence(agent_info: dict):
    agent_pref = agent_info["value2issue"]
    agent_reasons = agent_info['value2reason']
    hp_sentence = f'opponent highest priority is {agent_pref["High"]} because {remove_bad_chars(agent_reasons["High"])}'.replace(' I ', ' they ').replace(' me ', ' them ')
    mp_sentence = f'opponent medium priority is {agent_pref["Medium"]} because {remove_bad_chars(agent_reasons["Medium"])}'.replace(' I ', ' they ').replace(' me ', ' them ')
    lp_sentence = f'opponent lowest priority is {agent_pref["Low"]} because {remove_bad_chars(agent_reasons["Low"])}'.replace(' I ', ' they ').replace(' me ', ' them ')
    
    return f"{(hp_sentence + '.') if hp_sentence[-1]!='.' else hp_sentence} {(mp_sentence + '.') if mp_sentence[-1]!='.' else mp_sentence} {(lp_sentence + '.') if lp_sentence[-1]!='.' else lp_sentence}"


def make_write_split_data(data: list, outfile_pth: str):
    out_file = open(outfile_pth, 'w')
    out_file.write('input_seq,response,opt_pref_sentence\n')

    a1 = "mturk_agent_1"
    a2 = "mturk_agent_2"
    for dialogue in data:
        agent_1_context = get_context(dialogue["participant_info"][a1])
        agent_1_opponent_context = get_opponent_context_sentence(dialogue["participant_info"][a2])
        agent_2_context = get_context(dialogue["participant_info"][a2])
        agent_2_opponent_context = get_opponent_context_sentence(dialogue["participant_info"][a1])

        dialogue_1 = f'{agent_1_context}'
        dialogue_2 = f'{agent_2_context}'

        # First utterance must have <HISTORY> token appended before it
        text = dialogue['chat_logs'][0]['text']
        if text == 'Submit-Deal':
            sentence = get_submit_deal_sentence(dialogue['chat_logs'][1]['task_data'])
        elif text == 'Accept-Deal':
            sentence = ACCEPT_DEAL_SENTENCE
        elif text == 'Walk-Away':
            sentence = WALKAWAY_SENTENCE
        else:
            sentence = remove_bad_chars(text)
        id = dialogue['chat_logs'][0]['id']
        if id == a1:
            out_file.write(f'"{dialogue_1}","<YOU> {sentence}","{agent_1_opponent_context}"\n')
            dialogue_1 += f' <HISTORY> <YOU> {sentence}'
            dialogue_2 += f' <HISTORY> <THEM> {sentence}'
        elif id == a2:
            out_file.write(f'"{dialogue_2}","<YOU> {sentence}","{agent_2_opponent_context}"\n')
            dialogue_1 += f' <HISTORY> <THEM> {sentence}'
            dialogue_2 += f' <HISTORY> <YOU> {sentence}'
        else:
            raise Exception('INVALID AGENT ID')

        # Write rest of utterances in dialogue
        for c in dialogue['chat_logs'][1:]:
            if c['text'] == 'Submit-Deal':
                sentence = get_submit_deal_sentence(c['task_data'])
            elif c['text'] == 'Accept-Deal':
                sentence = ACCEPT_DEAL_SENTENCE
            elif c['text'] == 'Walk-Away':
                sentence = WALKAWAY_SENTENCE
            else:
                sentence = remove_bad_chars(c['text'])
            
            id = c['id']
            if id == a1:
                out_file.write(f'"{dialogue_1}","<YOU> {sentence}","{agent_1_opponent_context}"\n')
                dialogue_1 += f' <YOU> {sentence}'
                dialogue_2 += f' <THEM> {sentence}'
            elif id == a2:
                out_file.write(f'"{dialogue_2}","<YOU> {sentence}","{agent_2_opponent_context}"\n')
                dialogue_1 += f' <THEM> {sentence}'
                dialogue_2 += f' <YOU> {sentence}'
            else:
                raise Exception('INVALID AGENT ID')


def main():
    if os.path.exists(DESTINATION_DIR):
        raise FileExistsError('SPLIT DIRECTORY ALREADY EXISTS')
    else:
        os.mkdir(DESTINATION_DIR)

    if SPLIT_TRAIN + SPLIT_EVAL + SPLIT_TEST != 1:
        fix_train_test_split()

    data = json.load(open('casino.json'))
    random.seed(2022)
    random.shuffle(data)

    train_end = int(SPLIT_TRAIN*len(data))
    eval_end = int((SPLIT_TRAIN+SPLIT_EVAL)*len(data))

    make_write_split_data(data[:train_end], f'{DESTINATION_DIR}/train.csv')
    make_write_split_data(data[train_end:eval_end], f'{DESTINATION_DIR}/eval.csv')
    make_write_split_data(data[eval_end:], f'{DESTINATION_DIR}/test.csv')


if __name__ == '__main__':
    main()
