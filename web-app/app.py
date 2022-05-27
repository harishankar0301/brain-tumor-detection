import imp
from flask import Flask, render_template, request
import pickle
import os
import helpers
import extract_features as exf
app = Flask(__name__)

model = pickle.load(open('tuned_rfc.sav', 'rb'))
norm= pickle.load(open('norm_dict.pkl','rb'))

# UPLOAD_FOLDER = 'uploads'
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html', title='Home')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # add form input tag's name to data list
        f = request.files['file']  
        f.save(os.path.join('uploads',f.filename))  
        #f.save(f.filename)  
        im_path='uploads/'+f.filename
        im_name = f.filename
        #im_path= f.filename

        #Preprocessing
        preprocessing = helpers.Preprocessing()
        preprocessing.preproces(im_path)
        preprocessing.binarization()
        preprocessing.removingSkul()
        preprocessing.enhanceImage()
        preprocessing.segmentation()
        result=preprocessing.getInfectedRegion()

        #If no tumor found in preprocessing itself then skip classification
        if((result[0]==0 and result[1]==0 ) or (result[2]==0 and result[3]==0 )):
            text="Your Brain seems Healthy!"
            return render_template("index.html", output=0, text=text)
        
        else:


            df = exf.extract_features('tmp/'+im_name)

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
            
            output=model.predict(df)
            # Clean UP        
            #os.remove("tmp/"+im_name)
            os.remove(im_path)
            
            if(output):
                text="Your Brain may have tumor! "
                return render_template('index.html', output=output, text=text)
            else:
                text="Your Brain seems Healthy!"
                return render_template("index.html", output=output, text=text)
    
    
    return render_template("index.html")


if(__name__ == '__main__'):
    app.run()
