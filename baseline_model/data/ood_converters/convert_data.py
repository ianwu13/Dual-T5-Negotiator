import json
import re


def main():
    data = json.load(open('casino.json'))

    out_file = open('baseline_casino.csv', 'w')
    out_file.write('input_seq,response\n')

    a1 = "mturk_agent_1"
    a2 = "mturk_agent_2"
    for dialogue in data:
        dialogue_1 = ''
        dialogue_2 = ''
        for c in dialogue['chat_logs']:
            sentence = re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9\. ]+', '', c['text'])).strip()
            id = c['id']
            response = f'{sentence} <EOS>'
            if id == a1:
                if len(dialogue_1) != 0:
                    out_file.write(f'"{dialogue_1}","{response}"\n')
                    dialogue_1 += ' '
                else:
                    out_file.write(f'" ","{response}"\n')
                if len(dialogue_2) != 0:
                    dialogue_2 += ' '
                dialogue_1 += f'<YOU> {sentence}'
                dialogue_2 += f'<THEM> {sentence}'
            elif id == a2:
                
                if len(dialogue_2) != 0:
                    out_file.write(f'"{dialogue_2}","{response}"\n')
                    dialogue_2 += ' '
                else:
                    out_file.write(f'" ","{response}"\n')
                if len(dialogue_1) != 0:
                    dialogue_1 += ' '
                dialogue_1 += f'<THEM> {sentence}'
                dialogue_2 += f'<YOU> {sentence}'
            else:
                raise Exception('INVALID AGENT ID')


if __name__ == '__main__':
    main()
