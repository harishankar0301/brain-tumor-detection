import pickle
import helpers
import extract_features as exf


model = pickle.load(open('savedKnnModel.sav', 'rb'))
norm= pickle.load(open('norm_dict.pkl','rb'))
im_path='Y3.jpg'

       
preprocessing = helpers.Preprocessing()
preprocessing.preproces(im_path)
preprocessing.binarization()
preprocessing.removingSkul()
preprocessing.enhanceImage()
preprocessing.segmentation()
preprocessing.getInfectedRegion()
df = exf.extract_features('tmp/'+im_path)
print(df)
df.loc[0:,'Mean'] = (df['Mean']) / norm['Mean']
df.loc[0:,'ASM'] = (df['ASM']  / norm['ASM']) 
df.loc[0:,'contrast'] = (df['contrast']) / norm['contrast']
df.loc[0:,'correlation'] = (df['correlation']  /  norm['correlation']) 
df.loc[0:,'dissimilarity'] = (df['dissimilarity'] / norm['dissimilarity'])
df.loc[0:,'energy'] = (df['energy']  /  norm['energy'])
df.loc[0:,'kurtosis'] = (df['kurtosis'] /  norm['kurtosis'])
df.loc[0:,'skew'] = (df['skew']  / norm['skew'])
df.loc[0:,'Standard Deviation'] = (df['Standard Deviation']  / norm['Standard Deviation'])
df.loc[0:,'area'] = (df['area']) / norm['area']
df.loc[0:,'homogeneity'] = (df['homogeneity']  / norm['homogeneity']) 
df.loc[0:,'orientation'] = (df['orientation']  /  norm['orientation']) 
df.loc[0:,'convex_area'] = (df['convex_area'] / norm['convex_area'])
df.loc[0:,'eccentricity'] = (df['eccentricity'] /  norm['eccentricity'])
print(df)

print(model.predict(df))