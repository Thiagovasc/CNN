import pandas as pd

lesion_types: object = {
    'bcc': 'Carcinoma basocelular',
    'bkl': 'Ceratose benigna',
    'akiec': 'Ceratose actinica',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Nevos melanociticos',
    'vasc': 'Lesoes vasculares'
}
data = pd.read_csv('../data/HAM10000_metadata.csv')
data['lesion_types'] = data['dx'].map(lesion_types.get)
