#!/usr/bin/env python
# encoding: utf-8

"""
This is the flask part for launching plasmodesma web view

Laura Duciel
Inspired from PALMA web server.
contact: laura.duciel@casc4de.eu
"""

import os, glob, time, configparser
import os.path as op
import subprocess
from uuid import uuid4
from werkzeug import secure_filename
from flask import render_template, request, redirect, url_for, flash
from mail import SendMail
import codecs
from flask import Flask
import webbrowser,threading
from sys import platform as _platform

##########################################################
# CONFIGURATION
MAINTENANCE = False   # set to True for maintenance
PORT = 8000
ALLOWED_EXTENSIONS = set(['zip'])
#JOBSDIR = "/Volumes/biak_1ToHD/rdc/PALMA_WEB/jobs"  # location of the jobs to do
JOBSDIR = "/kiefb/delsuc/PALMA/JOBS"       # location of the jobs to do
#JOBSDIR = "/Users/mad/Documents/ mad/ publi/DOSY-W/new/server/jobs"       # location of the jobs to do
JOBMAX = 10                                # maximum 10 submission per hours
NPBPROCS = 16                               # number of procs for PALMA computation
app=Flask(__name__)
#app.config['UPLOAD_FOLDER'] = '/'
app.config['MAX_CONTENT_LENGTH'] = 20*1024*1024 #Uploaded file max size : 20 Mo 
app.debug = False
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT' #Needed for message flashing
##########################################################

if MAINTENANCE:
    maint_route = "/"
    home_route = "/maint.html"
else:
    maint_route = "/maint.html"
    home_route = "/"


@app.route(maint_route)
def maintenance():
    """Return the matinenance page, set to / in maintenance mode"""
    return render_template('maint.html',title='Home')

@app.route(home_route)
@app.route("/home.html")
def home():
    """Returns home.html"""
    return render_template('home.html',title='Home')
@app.route("/analysis.html")
def analysis():
    """Returns home.html"""
    return render_template('analysis.html',title='Data Analysis')

@app.route('/form.html', methods=['GET', 'POST'])
def form():
    """Returns form.html
    On POST request :
    - Returns an error page after trying to upload an incorrect type file, a file with a hazardous name or sending more than JOBMAX files/hour.
    - Manages data and parameters files upload and creates a dedicated folder for the job. 
    """
    if request.method == 'POST':
        email = request.form['email']
        DiffLimit = request.form['DiffLimits']
        Comment = request.form['Comment']
        print ("A ", email, DiffLimit, Comment)
        if userQuota(email):
            dataFile = request.files['file'] 
            if dataFile and allowed_file(dataFile.filename):
                jobId = str(uuid4()) #generates an unique job folder name
                os.mkdir(op.join(JOBSDIR, jobId))
#                os.mkdir("/kiefb/delsuc/PALMA/JOBS/titi")
                print ("B ", jobId, email, DiffLimit, Comment)
                paramFileInit(jobId, email, DiffLimit, Comment)
                filename = secure_filename(dataFile.filename)  
                dataFile.save(op.join(JOBSDIR,jobId, filename))
                flash('Successful upload !')
                flash('Starting the Job, jobid : %s ...' % (jobId,))
                curdir = os.getcwd()
                os.chdir(JOBSDIR)
                batchcmd = "sbatch -p surf /kiefb/delsuc/PALMA/PALMA_Job.sh %s" % (op.join(JOBSDIR, jobId),)
#                flash(batchcmd)
#                subprocess.call(batchcmd, shell=True)

#                batchcmd = """
#cp {0}/QM_Job_ref/* {1}/
#scp -r {1} palma@192.168.45.40:PALMA/JOBS/QM_qJobs
#""".format(JOBSDIR, op.join(JOBSDIR, jobId))
                #flash(batchcmd)

                job = subprocess.check_output(batchcmd, shell=True)
                flash(job)
                os.chdir(curdir)
                flash('...please wait for the results in your email box.')
#                return redirect(url_for('form'))
                return redirect(url_for('done'))
            else:
                return render_template('error.html',title='Invalid file type',error='Invalid file type')
        else:
            return render_template('error.html',title='Too many jobs submitted',error='Please do not submit more than %d jobs/hour'%JOBMAX)
    else:
        return render_template('form.html',title='Submit form')

@app.route('/HowTo.html')
def HowTo():
    """Returns How-To.html"""
    return render_template('HowTo.html',title='How-To')

@app.route('/done.html')
def done():
    """Returns done.html"""
    return render_template('done.html',title='Done..')

@app.route('/queue')
def queue():
    """Show the current state of server"""
    return render_template('queue.html',message=subprocess.check_output("squeue -u delsuc",shell=True) ) 
    
@app.route('/about.html')
def about():
    """Returns about.html"""
    return render_template('about.html',title='About')

@app.route('/contact.html',methods=['GET', 'POST'])
def contact():
    """Returns contact.html
    On POST request, sends text form content to a dedicated email adress
    """
    if request.method == 'POST':
        #Please change the adress
        SendMail(to='madelsuc@unistra.fr',subject='User mail from DOSY Analysis Service',text=request.form['mailContent'])
        flash('Message sent. Thanks for the feedback !')
        return redirect(url_for('contact'))
    return render_template('contact.html',title='Contact us')
    
def userQuota(email):
    """Returns false if more than JOBMAX jobs with current request's email were already submitted in the last hour"""
    config=ConfigParser.ConfigParser()
    token=0
    for files in glob.glob(op.join(JOBSDIR,'*','parameters.txt')):
        if os.stat(files).st_ctime>=(time.time()-(60*60)):
            config.read(files)
            if email==config.get('USER_INFO','email'):
                token+=1
    if token>=JOBMAX: #JOBMAX jobs/hour limit
        return False
    else:
        return True
def allowed_file(filename):
    """Tests if the uploaded file extension is in global variable ALLOWED_EXTENSIONS"""
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS             
def paramFileInit(jobId, email, difflim, comment):
    """Creates parameters.txt in the current job folder and writes in the parameters sent by user.
    /!\ Parameters are entered as an unsorted dictionary, so using configparser to retrieve them is prefered."""
    print ("preparing Job ", jobId, email, difflim, comment)
    # difflim codes the limits we're going to use, see form.html for details
    if difflim == "1":    # standard
        dmin = 50
        dmax = 10000
        lamda = 0.05
    elif difflim == "2":  # light
        dmin = 50
        dmax = 10000
        lamda = 0.01
    elif difflim == "3":    # large
        dmin = 10
        dmax = 10000
        lamda = 1.0
    elif difflim == "4":    # very large
        dmin = 1
        dmax = 10000
        lamda = 1.0
    else:
        dmin = 50
        dmax = 10000
        lamda = 0.1
    with codecs.open(op.join(JOBSDIR,jobId,'parameters.txt'), 'w', "utf_8") as configfile:
        configfile.write('[USER_INFO]\n')
        configfile.write("email = {}\n".format(email))
        configfile.write('\n[PARAM]\n')
        configfile.write("dmin = {:f}\n".format(dmin))
        configfile.write("dmax = {:f}\n".format(dmax))
        configfile.write("nbprocs = {:d}\n".format(NPBPROCS))
        configfile.write("lambda = {:f}\n".format(lamda))
        configfile.write(u"comment = {:s}\n".format(comment))

def lauch_computation(jobId):
    pass

def main(startweb=True):
    port = PORT
    #url = "http://localhost:{0}".format(port) # 127.0.0.1
    url = "http://127.0.0.1:{0}/".format(port) # 127.0.0.1
    if startweb:
        if _platform == "linux" or _platform == "linux2":
            platf = 'lin'
        elif _platform == "darwin":
            platf = 'mac'
        elif _platform == "win32":
            platf = 'win'
        # MacOS
        if platf == 'mac':
            chrome_path = 'open -a /Applications/Google\\ Chrome.app %s'
        # Linux
        if platf == 'lin':
            chrome_path = '/usr/bin/google-chrome %s'
        if platf != 'win':
            b = webbrowser.get(chrome_path)
            threading.Timer(1.25, lambda: b.open_new(url)).start() # open a page in the browser.
        else:
            webbrowser.open(url, new=0, autoraise=True)
            # subprocess.Popen('"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" {0}'.format(url))
    print( """
    ***************************************************************
    Launching the Plasmodesma program

    If the Chrome browser does not show up or is not available,
    open your favorite browser and go to the following addess:

    {0}

    """.format(url) )
    try:
        app.run(port = port , host='127.0.0.1') # port
    except OSError:
        print("################################################################################")
        print("             WARNING\n")
        print("The program could not be started, this could be related to a PORT already in use")
        print("Please change the port number in the configuration file and retry.\n")
        print("################################################################################")


if __name__ == '__main__':
    main()