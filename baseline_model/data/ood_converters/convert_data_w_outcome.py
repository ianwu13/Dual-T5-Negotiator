import json
import re


def main():
    data = json.load(open('casino.json'))

    out_file = open('baseline_casino_w_outcome.csv', 'w')
    out_file.write('input_seq,response\n')

    a1 = "mturk_agent_1"
    a2 = "mturk_agent_2"
    for dialogue in data:
        dialogue_1 = ''
        dialogue_2 = ''
        for c in dialogue['chat_logs']:
            if c['text'] == 'Submit-Deal':
                task_dat_you = c['task_data']['issue2youget']
                task_dat_them = c['task_data']['issue2theyget']
                sentence = f"<SUBMIT> {task_dat_you['Food']} {task_dat_you['Water']} {task_dat_you['Firewood']} {task_dat_them['Food']} {task_dat_them['Water']} {task_dat_them['Firewood']}"
            elif c['text'] == 'Accept-Deal':
                sentence = '<ACCEPT>'
            elif c['text'] == 'Walk-Away':
                sentence = '<WALKAWAY>'
            else:
                sentence = re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9\. ]+', '', c['text'])).strip()
            
            id = c['id']
            if id == a1:
                if len(dialogue_1) != 0:
                    out_file.write(f'"{dialogue_1}","{sentence} <EOS>"\n')
                    dialogue_1 += ' '
                else:
                    out_file.write(f'" ","{sentence} <EOS>"\n')
                if len(dialogue_2) != 0:
                    dialogue_2 += ' '
                dialogue_1 += f'<YOU> {sentence}'
                dialogue_2 += f'<THEM> {sentence}'
            elif id == a2:
                
                if len(dialogue_2) != 0:
                    out_file.write(f'"{dialogue_2}","{sentence} <EOS>"\n')
                    dialogue_2 += ' '
                else:
                    out_file.write(f'" ","{sentence} <EOS>"\n')
                if len(dialogue_1) != 0:
                    dialogue_1 += ' '
                dialogue_1 += f'<THEM> {sentence}'
                dialogue_2 += f'<YOU> {sentence}'
            else:
                raise Exception('INVALID AGENT ID')


if __name__ == '__main__':
    main()
