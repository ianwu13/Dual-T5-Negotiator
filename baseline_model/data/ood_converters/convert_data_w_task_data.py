import json
import re


def get_agent_data(agent_info: dict):
    agent_pref = agent_info["value2issue"]
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
    return f"{agent_pref['High']} {agent_pref['Medium']} {agent_pref['Low']} {agent_personality['svo']}{big_five}"


def main():
    data = json.load(open('casino.json'))

    out_file = open('baseline_casino_w_task_data.csv', 'w')
    out_file.write('input_seq,response\n')

    a1 = "mturk_agent_1"
    a2 = "mturk_agent_2"
    for dialogue in data:
        agent_data_1 = get_agent_data(dialogue["participant_info"][a1])
        agent_data_2 = get_agent_data(dialogue["participant_info"][a2])

        dialogue_1 = f'{agent_data_1}'
        dialogue_2 = f'{agent_data_2}'
        for c in dialogue['chat_logs']:
            sentence = re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9\. ]+', '', c['text'])).strip()
            id = c['id']
            response = f'{sentence} <EOS>'
            if id == a1:
                out_file.write(f'"{dialogue_1}","{response}"\n')
                dialogue_1 += f' <YOU> {sentence}'
                dialogue_2 += f' <THEM> {sentence}'
            elif id == a2:
                out_file.write(f'"{dialogue_2}","{response}"\n')
                dialogue_1 += f' <THEM> {sentence}'
                dialogue_2 += f' <YOU> {sentence}'
            else:
                raise Exception('INVALID AGENT ID')


if __name__ == '__main__':
    main()
