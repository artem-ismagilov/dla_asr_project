from editdistance import distance

def calc_cer(target_text, predicted_text) -> float:
    if target_text == '':
        return 1
    return distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == '':
        return 1
    t = target_text.split()
    return distance(t, predicted_text.split()) / len(t)
