import json
import re

ACCEPT_DEAL_SENTENCE = 'this deal is acceptable.'
WALKAWAY_SENTENCE = 'i am walking away from this deal'


def get_submit_deal_sentence(task_data):
    task_dat_you = task_data['issue2youget']
    task_dat_them = task_data['issue2theyget']
    return f"lets submit this deal. i get {task_dat_you['Food']} food {task_dat_you['Water']} water and {task_dat_you['Firewood']} firewood. you get {task_dat_them['Food']} food {task_dat_them['Water']} water and {task_dat_them['Firewood']} firewood."


def remove_bad_chars(input: str):
    out = re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9\.\!\? ]+', '', input)).strip()
    return out if len(out) > 0 else " "


def get_context(agent_info: dict):
    agent_pref = agent_info["value2issue"]
    agent_reasons = agent_info['value2reason']
    hp_sentence = f'my highest priority is {agent_pref["High"]} because {remove_bad_chars(agent_reasons["High"])}'
    mp_sentence = f'my medium priority is {agent_pref["Medium"]} because {remove_bad_chars(agent_reasons["Medium"])}'
    lp_sentence = f'my lowest priority is {agent_pref["Low"]} because {remove_bad_chars(agent_reasons["Low"])}'
    
    return f'<CONTEXT> {hp_sentence} {mp_sentence} {lp_sentence}'


def get_opponent_pref(agent_info: dict):
    agent_pref = agent_info["value2issue"]
    return f'<CONTEXT> high {agent_pref["High"]} medium {agent_pref["Medium"]} low {agent_pref["Low"]}'


def main():
    data = json.load(open('casino.json'))

    out_file = open('baseline_casino_opponent_pref.csv', 'w')
    out_file.write('input_seq,response\n')

    a1 = "mturk_agent_1"
    a2 = "mturk_agent_2"
    for dialogue in data:
        agent_1_context = get_context(dialogue["participant_info"][a1])
        agent_2_context = get_context(dialogue["participant_info"][a2])

        opponent_pref_1 = get_opponent_pref(dialogue["participant_info"][a2])
        opponent_pref_2 = get_opponent_pref(dialogue["participant_info"][a1])

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
            out_file.write(f'"{dialogue_1}","{opponent_pref_1} <YOU> {sentence}"\n')
            dialogue_1 += f' <HISTORY> <YOU> {sentence}'
            dialogue_2 += f' <HISTORY> <THEM> {sentence}'
        elif id == a2:
            out_file.write(f'"{dialogue_2}","{opponent_pref_2} <YOU> {sentence}"\n')
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
                out_file.write(f'"{dialogue_1}","{opponent_pref_1} <YOU> {sentence}"\n')
                dialogue_1 += f' <YOU> {sentence}'
                dialogue_2 += f' <THEM> {sentence}'
            elif id == a2:
                out_file.write(f'"{dialogue_2}","{opponent_pref_2} <YOU> {sentence}"\n')
                dialogue_1 += f' <THEM> {sentence}'
                dialogue_2 += f' <YOU> {sentence}'
            else:
                raise Exception('INVALID AGENT ID')


if __name__ == '__main__':
    main()
