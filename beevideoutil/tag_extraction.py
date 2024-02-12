#https://stackoverflow.com/a/51950538

from datetime import datetime, timedelta
from shutil import copyfile
from multiprocessing import Process, Queue
import sys
import re
import pandas as pd
import numpy as np

import signal
import io
import os
import shutil
import subprocess
from os.path import join, dirname, isfile
import threading
import time
from time import strftime, localtime

import ctypes
def _set_pdeathsig(sig=signal.SIGTERM):
    """help function to ensure once parent process exits, its childrent processes will automatically die
       See https://stackoverflow.com/a/43152455
    """
    def callable():
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig)
    return callable

### LOCAL TAGS COMPUTATION

def run_tags(item, mp4root, tagsroot=None, f0=0, f1=None, *, jobdir, nthreads=4):
    # assume the following fields are available in item (pd.Series)
    # mp4file, tagsfile
    mp4fullfile = os.path.join(mp4root,item.mp4file)

    tagfilename = item.tagsfile.split("/")[-1]
    if (tagsroot is not None):
        tagsfullfile = os.path.join(tagsroot,item.tagsfile)
    else:
        tagsfullfile = None

    #logfile = os.path.join(jobdir,'log.txt')
    if (f1 is None):
        f1 = int(item.frames)-1
    #outjson = os.path.join(jobdir,'tags_{:05}-{:05}.json'.format(0,f1))
    outjson = tagfilename
    outfullfile = os.path.join(jobdir, tagfilename)

    cmd = [#'echo', 
            'apriltagdetect.py',
            '-V', mp4fullfile, '-outdir', jobdir,
            '-F', 'tag25h5inv',
            '-f0', str(int(f0)), '-f1', str(int(f1)), '-1', '-D=0', '-fps', '20.0', 
            '-t', str(nthreads),  # 4 worker pool in apriltag
            '-m', '-mv', '2', '-2', '-x', '2', '-D', '256',
            '-rgb_mean',
            '-cvt', str(nthreads),
            '-outjson', outjson,
            '-progress']

    print("mp4fullfile ",mp4fullfile)
    print("jobdir      ",jobdir)
    #print("logfile     ",logfile)
    print("outjson     ",outjson)
    print("tagsfullfile",tagsfullfile)
    print("f0,f1       ",(f0,f1))
    print("cmd         ", " ".join(cmd))

    print("RUNNING...")
    print("BEGIN @ {}\n".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    then = datetime.now()
    my_env = os.environ.copy()
    my_env["OMP_NUM_THREADS"] = "4"
    try:
             
            correct = subprocess.run(cmd, env=my_env,
                                     check=True, text=True, stdout=sys.stdout, stderr=subprocess.STDOUT,
                                     preexec_fn=_set_pdeathsig(signal.SIGTERM))
        
    except subprocess.CalledProcessError as e:
        sys.stdout.flush()
        print("ERROR @ {}: RETURN CODE = {}\n".format(datetime.now().strftime("%Y%m%d-%H%M%S"), e.returncode))
        print("ERROR running apriltag subprocess: error code ",e.returncode)
        raise
    sys.stdout.flush()
    print("apriltag done @ {}\n".format(datetime.now().strftime("%Y%m%d-%H%M%S")))

    print("Moving...")
    if (tagsroot is not None):
        print("Move outjson {} to {}".format(outjson,tagsfullfile))
        copyfile(outfullfile, tagsfullfile)
    else:
        print("outfile not moved")

    print("DONE @ {}\n".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    now = datetime.now()
    duration = (now - then).total_seconds()/60
    print(f"TOTAL ELAPSED {duration:.2f} min = {duration/60:.2f} h")

    return


class JobTags(Process):
    counter = 1
    def __init__(self, jobroot, label="default", args=[], kwargs={}, cpu_affinity=None, nthreads=4):
        self.args = args
        self.kwargs = kwargs
        
        self.jobroot = jobroot
        self.jobid = JobTags.counter
        JobTags.counter += 1
        
        jobname="tags-job{}___{}___{}".format(self.jobid,
                                         datetime.now().strftime("%Y%m%d-%H%M%S"),
                                         label.replace('/','--'))
        jobdir=os.path.join(self.jobroot,jobname)
        
        self.jobdir = jobdir
        self.logfile = os.path.join(jobdir,'log.txt')
        
        #print(f'Logging to {self.logfile}')
        
        self.start_time=None
        self.end_time=None
        
        self.cpu_affinity=cpu_affinity
        self.nthreads=nthreads
        
        self.exitCallbacks=[]
        self.exitThread=None
        
        super(JobTags, self).__init__()
    
    def startWithCallback(self,cb):
        def _startWaitCallbacks(self):
            self.start()
            self.join()
            self.end_time=datetime.now()
            for cb in self.exitCallbacks:
                cb(self)
        self.exitCallbacks=[cb]
        self.exitThread = threading.Thread(target=_startWaitCallbacks, args=(self,))
        self.exitThread.start() # Call process.start inside, and wait for it
    
    def start(self):
        self.start_time=datetime.now()
        super(JobTags, self).start()
        
    def set_cpu_affinity(self, cpu_affinity):
        self.cpu_affinity=cpu_affinity
        
        if (self.cpu_affinity is not None):
            #print('JobTags set cpu_affinity to',self.cpu_affinity,' pid=',os.getpid())
            print('JobTags set cpu_affinity to',self.cpu_affinity,' pid=',self.pid)
            output = subprocess.check_output("taskset -a -p -c '{}' {}".format(self.cpu_affinity, os.getpid()), shell=True)
            print(output.decode())
            #os.system("taskset -p -c {} {}".format(self.cpu_affinity, os.getpid()))
            #os.system("taskset -p {}".format(os.getpid()))
        
    def elapsed(job):
        if (job.start_time is None):
            return -1
        now = datetime.now()
        duration = (now - job.start_time).total_seconds()
        return duration
    
    def run(self):
        # Define the logging in run(), MyProc's entry function when it is .start()-ed 
        #     p = MyProc()
        #     p.start()
        
        os.makedirs(self.jobdir,exist_ok=False)
        self.initialize_logging()
        
        if (self.cpu_affinity is not None):
            print('JobTags set cpu_affinity to',self.cpu_affinity,' pid=',os.getpid())
            output = subprocess.check_output("taskset -p -c '{}' {}".format(self.cpu_affinity, os.getpid()), shell=True)
            print(output)
            #os.system("taskset -p -c {} {}".format(self.cpu_affinity, os.getpid()))
            #os.system("taskset -p {}".format(os.getpid()))

        print('JobTags run_tags:')

        start_time = datetime.now()
        print("START TIME",start_time.strftime("%Y%m%d-%H%M%S"), flush=True)
        
        def handler(s,f):
            raise Exception("Received SIGTERM. Interrupting.")
        signal.signal(signal.SIGTERM, handler)
        
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            run_tags(*self.args, **self.kwargs, jobdir=self.jobdir, nthreads=self.nthreads)
        except:
            end_time = datetime.now()
            sys.stdout.flush()
            sys.stderr.flush()
            print("END TIME",end_time.strftime("%Y%m%d-%H%M%S"),"ABORTED", file=sys.stderr)
            duration = (end_time - start_time).total_seconds()/60
            print(f"TOTAL ELAPSED {duration:.2f} min = {duration/60:.2f} h", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            raise
        end_time = datetime.now()
        sys.stdout.flush()
        sys.stderr.flush()
        print("END TIME",end_time.strftime("%Y%m%d-%H%M%S"), flush=True)
        duration = (end_time - start_time).total_seconds()/60
        print(f"TOTAL ELAPSED {duration:.2f} min = {duration/60:.2f} h", flush=True)

    def initialize_logging(self):
        #print(f'Setting up loggin to {self.logfile}')

        # open stdout in binary mode, then wrap it in a TextIOWrapper and enable write_through
        #sys.stdout = io.TextIOWrapper(open(self.logfile, "wb", buffering=0), write_through=True)
        # for flushing on newlines only use :
        #sys.stdout.reconfigure(line_buffering=True)
    
        sys.stdout = open(self.logfile, "w", buffering=1)
        sys.stderr = sys.stdout

        #print(f'stdout/stderr initialized to {self.logfile}')
        
    def log(self, show=True, tail=0, head=0, logfile=None):
        # Show log for Job
        if (logfile is None):
            logfile = self.logfile
        else:
            logfile = os.path.join(self.jobdir,logfile)
        with open(logfile,'r') as log:
            if (tail>0):
                lines = log.readlines()
                lines = ''.join(lines[-tail:])
            elif (head>0):
                lines = log.readlines(head)
                lines = ''.join(lines[-tail:])
            else:
                lines = log.read()
                
        if (show): 
            print(lines)
        else:
            return(lines)
        
    def tagtail(job, tail=12, show=True):
        # Return text of generated tag file
        item=job.args[0]
        tagname=item.tagsfile.split('/')[-1]
        tagfile = os.path.join(job.jobdir,f"{tagname}.tmp")

        try:
            with open(tagfile,'r') as log:
                if (tail>0):
                    lines = log.readlines()
                    lines = ''.join(lines[-tail:])
                else:
                    lines = log.read()
        except:
            tagfile = os.path.join(job.jobdir,f"{tagname}")
            try:
                with open(tagfile,'r') as log:
                    if (tail>0):
                        lines = log.readlines()
                        lines = ''.join(lines[-tail:])
                    else:
                        lines = log.read()
            except:
                lines = ''

        if (show): 
            print(lines)
        else:
            return(lines)

    def progress(job):
        # Parse generated tag file for last computed frame number
        tail = job.tagtail(show=False)
        out=re.search('\n  "([0-9]+)":\{"tags":\[\n', tail)
        if (out is None):
            return -1
        else:
            return int(out.group(1))

        
def jobstatus(jobs, tagsdf=None):
    jobdf = pd.DataFrame(jobs,columns=['job'])
    def get_end_time(job):
        if (job.end_time is not None):
            return job.end_time
        # Parse generated tag file for last computed frame number
        tail = job.log(tail=2,show=False)
        out=re.search('END TIME ([0-9][0-9][0-9][0-9])([0-9][0-9])([0-9][0-9])-([0-9][0-9])([0-9][0-9])([0-9][0-9])\n', tail)
        #print(out.group(0))
        if (out is None):
            return -1
        else:
            YY,MM,DD,hh,mm,ss=[int(out.group(i)) for i in range(1,7)]
            job.end_time=datetime(YY,MM,DD,hh,mm,ss)
            return job.end_time
    def get_total_elapsed(job):
        end_time = get_end_time(job)
        start_time = job.start_time
        #print(start_time, end_time)
        if (end_time is -1):
            return job.elapsed()
        else:
            duration = (end_time - start_time).total_seconds()
            return duration
        
    def progress(job):
        tail = job.tagtail(show=False)
        #out=re.findall('\n  "([0-9]+)":\{"tags":\[\n', tail)
        allout = [i for i in re.finditer('\n  "([0-9]+)":\{"tags":\[\n',tail)]
        #print(allout)
        if (len(allout)==0):
            out=None
        else:
            out=allout[-1]
        if (out is None):
            return 0
        else:
            return int(out.group(1))+1
    def elapsed(job):
        if ((not hasattr(job, 'start_time')) or (job.start_time is None)):
            return np.nan
        now = datetime.now()
        duration = (now - job.start_time).total_seconds()
        return duration
    def getinfo(item):
        job=item.job
        prog = int(progress(job))
        elap = get_total_elapsed(job)
        fps = prog/elap
        mp4file=job.args[0].mp4file
        total=-1
        if (tagsdf is not None):
            total = tagsdf[tagsdf.mp4file==mp4file].frames.iloc[0]
        est_hh = np.float64(np.inf)
        if (prog>0):
            est_hh = np.float64(72000)/prog*elap/3600
        if (np.isnan(est_hh) or np.isinf(est_hh)):
            est_end = 'unknown'
        else:
            est_end = (job.start_time+timedelta(hours=est_hh)).strftime('%H:%M')
        if (job.end_time is not None):
            est_end = "Done "+est_end
        else:
            est_end = "Est  "+est_end
        return pd.Series(dict(job=job, pid=job.pid, jobid=f"job{job.jobid}", 
                    mp4=mp4file, progress=prog, total=total, perc=int(prog/total*100),
                              elapsed_mm=elap/60, elapsed_hh=elap/3600, proc_fps=fps, est_hh=est_hh,
                              est_end=est_end,
                             cpus=job.cpu_affinity))
    df = jobdf.apply(getinfo,axis=1)
    #pd.DataFrame([(job, job.pid, f"job{job.jobid}", job.args[0].mp4file, prog, elap/60, fps) 
    #                   for i,job in enumerate(jobs)], 
    #                  columns='job,pid,jobid,mp4,progress,elapsed-mm,proc-fps'.split(','))
    def formatjob(job):
        return repr(job).replace("<","").replace(">","")
    return df #df.style.format({'job':formatjob,'est_hh': '{:0.2f}'})


def runTags(item, aff, jobs, jobroot, mp4root, tagsroot, out):
    def jobDone(job):
         job.theJobIsDone=True
         out.append_stdout(f"Job finish: {job} {job.args[0].mp4file}  {datetime.now().strftime('%H:%M')}\n")
    with out:
        job = JobTags(jobroot, label=item.mp4file[:-4], 
                        args=(item,mp4root, tagsroot), kwargs=dict(f1=None), cpu_affinity=aff, nthreads=1)
                        #args=(item,mp4root, None), kwargs=dict(f1=2))
        #job.start()
        job.startWithCallback(jobDone)
        jobs.append(job)
        print(f"Job start:  {job} {job.args[0].mp4file}  {datetime.now().strftime('%H:%M')}")
    return job


import threading
import subprocess

def startWithCallback(process, onExit):
    """
    Call process.start(), and then calls the function onExit when the
    subprocess completes.

    onExit is a callable object, will receive process as parameter.
    """
    def runInThread(process, onExit):
        process.join()
        onExit(process)
        return

    thread = threading.Thread(target=runInThread,
                              args=(process, onExit))
    
    process.start()
    thread.start()

    return thread # returns immediately after the thread starts


import ipywidgets as widgets
from ipywidgets import Button, Label, HBox, VBox, Layout
from IPython.display import display, clear_output

def jobsGui(jobs):
    out = widgets.Output()

    def clicked_log(job,label):
        with out:
              # what happens when we press the button
              clear_output()
              job.log(tail=20)

    def clicked_tags(job,label):
        with out:
              # what happens when we press the button
              clear_output()
              job.tagtail()

    def clicked_update(jobs,progs):
        for i,job in enumerate(jobs):
            progs[i].value=job.progress()
            progs[i].description=str(job.progress())
            
    def clear_out():
        with out:
            clear_output()

    def adapt(cb, *args):
        def fun(_):
            cb(*args)
        return fun

    update_button=Button(description='Update',layout=widgets.Layout(width="100px"))
    clear_button=Button(description='Clear',layout=widgets.Layout(width="100px"))
    clear_button.on_click(adapt(clear_out))

    items=[]
    labels={}
    buttons={}
    progs=[]
    for i,j in enumerate(jobs):
        label=Label(value=f"{i} {j.args[0].mp4file}")
        #buttons[i]=Button(description=str(j),layout=Layout(width='300px'))
        button=Button(description='Log',layout=widgets.Layout(width="100px"))
        b = Button(description='Tagsfile',layout=widgets.Layout(width="100px"))
        b.on_click(adapt(clicked_tags,j,label))

        prog = widgets.IntProgress(
            value=0,
            min=0,
            max=72000,
            bar_style='info',
            style={'bar_color': '#00cf00'},
            orientation='horizontal'
        )
        progs.append(prog)
        button.on_click(adapt(clicked_log,j,label))

        items = items + [label,button, b, prog]

    update_button.on_click(adapt(clicked_update,jobs,progs))

    grid=widgets.GridBox(items, layout=widgets.Layout(grid_template_columns="max-content max-content max-content max-content"))

    #VBox([ HBox([label,button]), out ])

    display(VBox([HBox([update_button,clear_button]),grid,out]))
    
    
### REMOTE SLURM TAGS COMPUTATION
    
import os
from os.path import join
from tqdm.autonotebook import tqdm
import time

# conda install bcrypt paramiko pynacl
# ##pip3 install slurmqueen
# git clone git@github.com:rmegret/SlurmQueen.git
# pip3 install -e SlurmQueen
import slurmqueen

def local_execute(cmd, export=True):
    res=''
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            if (export):
                res+=line
            else:
                print(line, end='') # process line here
    return res

class SlurmJobTag():
    def __init__(self, config, item, mp4root, time='2:00', cpus_per_task=1, nthreads=1, name=None):
        self.starttime=strftime("%Y%m%d_%H%M%S", localtime())
        self.config=config
        self.item=item
        sani=re.sub('\.mp4$', '', item.mp4file.replace('/','++'))
        
        if (name is None):
            self.name=f'tags__{sani}__{self.starttime}'
        else:
            self.name=name
        
        #print(f"Setting up {self.name}")
        
        self.local_dir=join(config.local_directory,self.name)
        self.remote_dir=join(config.remote_directory,self.name)
        self.outjson=item.tagsfile.replace('/','++')
        self.remote_tags=join(self.remote_dir, self.outjson)
        self.local_tags=join(self.local_dir, self.outjson)
        
        # global mp4root
        self.mp4root = mp4root
        self.local_mp4 = join(self.mp4root, item.mp4file)
        self.remote_mp4root = '/work/rmegret/rmegret/slurmqueentest/mp4cache/'
        self.remote_mp4 = join(self.remote_mp4root, item.mp4file)
        self.remote_mp4dir = os.path.dirname(self.remote_mp4)

        self.f1=item.frames-1
        self.nthreads=nthreads
        
        self.cmd = f'apriltagdetect.py -V "{self.remote_mp4}" '\
               f' -outdir "{self.remote_dir}" -outjson "{self.outjson}"'\
               f' -f0 0 -f1 {int(self.f1)} -1 -D=0 -fps 20.0 '\
               f' -t {self.nthreads} -cvt {self.nthreads} '\
               f' -F tag25h5inv -m -mv 2 -2 -x 2 -D 256 '\
               f' -rgb_mean -progress '
        
        self.exp = slurmqueen.SlurmExperiment(
            self.name,
            *[ self.cmd ],
            #changing_args=[],
            dependencies=[],
            setup_commands=f'''
export PATH=/work/rmegret/rmegret/utils/swatbotics_apriltag/python/:"$PATH"
echo PATH=$PATH
echo PID=$$
set -x
which python
which apriltagdetect.py
ls -l {self.remote_mp4}  # MP4
top -b -n 1 | head -20   # TOP
mpstat -P ALL            # CPU usage
set +x
date --rfc-3339=seconds
'''
        )
        self.instance = self.exp.slurm_instance(config, debug_script=False)
        
        self.time=time
        self.cpus_per_task=cpus_per_task
    def jobid(self):
        return self.instance._jobid
    
    def __repr__(self):
        return f"SlurmJobTag[jobid={self.jobid()},mp4={self.item.mp4file}]"
        
    # VIDEO CACHE HANDLING
        
    def transfer_callback(self, pbar):
        def get_current_time_milli():
            return int(round(time.time() * 1000))
        def debounce(callback, throttle_time_limit=1000):
            # https://stackoverflow.com/question/31922875/throttle-a-function-call-in-python
            last_millis = get_current_time_milli()  
            def throttle(*args,**kwargs):
                nonlocal last_millis
                curr_millis = get_current_time_milli()
                if (curr_millis - last_millis) > throttle_time_limit:
                    last_millis = get_current_time_milli()
                    callback(*args,**kwargs)
            return throttle    
        def update_to(current, total):
            pbar.total = total
            pbar.n = current
            pbar.refresh()
        return debounce(update_to)
        
    def put_video(self, force=False, quiet=False):
        #self.config.server.execute(f'mkdir -p {self.remote_mp4root}; ls -ld {self.remote_mp4root}')
        self.config.server.execute(f'mkdir -p {self.remote_mp4dir}; ls -ld {self.remote_mp4dir}')
        
        if (self.check_video()):
            if (force):
                print(f'Existing MP4 {self.remote_mp4}. Forcing upload')
            else:
                print(f'Existing MP4 {self.remote_mp4}. Skipping upload')
                return
        if (quiet):
            with self.config.server.ftp_connect() as ftp:
                ftp.put(self.local_mp4,self.remote_mp4)
        else:
            print(f'Uploading to {self.remote_mp4}')
            with self.config.server.ftp_connect() as ftp:
                S=os.stat(self.local_mp4)
                with tqdm(total=S.st_size,mininterval=0.5,maxinterval=1.0,unit="B",unit_scale=True) as pbar:
                    ftp.put(self.local_mp4,self.remote_mp4, callback=self.transfer_callback(pbar))
                    pbar.update(pbar.total)
        
    def check_video(self):
        #self.config.server.execute(f'ls -l {self.remote_mp4}')
        remote_size = self.config.server.execute(f'stat --printf="%s" {self.remote_mp4}')
        if (remote_size==''):
            remote_size=-1
        else:
            remote_size=int(remote_size)
        local_size = int(os.stat(self.local_mp4).st_size)
        return local_size==remote_size
    
    def delete_video(self, only_if_complete=True, dryrun=False):
        if (self.check_tags(check_tmp=False, check_valid=True) != 'Complete'):
            print(f"Skipped {self.remote_mp4}. Not complete")
            return
                
        if (self.check_video()):
            if (dryrun):
                print(f"Delete {self.remote_mp4}  (DRY RUN)")
                return
            else:
                print(f"Delete {self.remote_mp4}")
                self.config.server.execute(f"rm {self.remote_mp4}")
        else:
            print(f"Skipped {self.remote_mp4}. File does not exist")

    # TAG HANDLING
    
    def get_tags(self, tmp=False, quiet=False, gather=False):
        
        if (tmp):
            remote_file = self.remote_tags+'.tmp'
            local_file = self.local_tags+'.tmp'
        else:
            remote_file = self.remote_tags
            local_file = self.local_tags
        
        if (gather):
            self.instance._gather()
        
        try:
            with self.config.server.ftp_connect() as ftp:
                if (not quiet):
                    print(remote_file, "=>", local_file)
                ftp.get(remote_file, local_file)
        except FileNotFoundError:
            print(f'Error: Remote file Not found {remote_file}. Skipped')
    
    def store_tags(self, tagsroot, force=False, dryrun=False, quiet=False):
        
        local_file = self.local_tags
        tagsfile = join(tagsroot,self.item.tagsfile)
        
        def qprint(*args,**kwargs):
            if (not quiet or dryrun):
                print(*args,**kwargs)
        
        if (not isfile(local_file)):
            self.get_tags(tmp=False, quiet=quiet, gather=False)
            if (not isfile(local_file)):
                qprint(f'Missing file, Skipped {local_file}, jobid={self.jobid()}')
                return 
        if (not self.check_valid_tagfile()):
            qprint(f'File incomplete, Skipped {tagsfile}, jobid={self.jobid()}')
            return
        if (isfile(tagsfile)):
            if (not force):
                qprint(f'File exist, Skipped {tagsfile}, jobid={self.jobid()}')
                return
            else:
                qprint(f'File exist, Force copy to {tagsfile}, jobid={self.jobid()}')
        else:
            qprint(f"Copy to {tagsfile}, jobid={self.jobid()}")
            
        if (not dryrun):
            os.makedirs(os.path.dirname(tagsfile), exist_ok=True)
            shutil.copy2(local_file, tagsfile)
            
    def stored(self, tagsroot):
        tagsfile = join(tagsroot,self.item.tagsfile)
        return isfile(tagsfile)
            
    def check_valid_tagfile(self):
        tail = local_execute(f'tail -10 "{self.local_tags}"',export=True)
        if (tail[-8:] == ']}\n}\n\n}\n'):
            return True
        else:
            return False
    
    def check_tags(self, check_tmp=False, check_valid=False):
        #self.config.server.execute(f'ls -l {self.remote_mp4}')
        remote_size = self.config.server.execute(f'stat --printf="%s" {self.remote_tags}')
        if (remote_size==''):
            if (check_tmp):
                remote_size = self.config.server.execute(f'stat --printf="%s" {self.remote_tags}.tmp')
                if (remote_size==''):
                    return 'NoFile'
                else:
                    return 'TmpFile'
            else:
                return 'NoFile'
        if (check_valid):
            tail = self.tail_tags(show=False)
            if (tail[-8:] == ']}\n}\n\n}\n'):
                return 'Complete'
            else:
                return 'Incomplete'
        else:
            return 'HasFile'
    
    def tail_tags(self, tmp=False, show=True):
        if (tmp):
            remote_file = self.remote_tags+".tmp"
        else:
            remote_file = self.remote_tags
        res = self.config.server.execute(f'tail -20 "{remote_file}"')
        if (show):
            print(res)
        else:
            return res
    def sacct(self):
        df=sacct([self], self.config.server, show=False)
        assert df.shape[0]==1, f"Expected 1 slurm task for job {self}, got {df.shape[0]}"
        return df.iloc[0]
    def progress(self):
        status = self.check_tags(check_tmp=True)
        if (status == 'TmpFile'):
            tail = self.tail_tags(tmp=True, show=False)
        elif (status in ['HasFile']):
            tail = self.tail_tags(tmp=False, show=False)
        else:
            return (0, 0)
        frames=int(self.item.frames)
        allout = [i for i in re.finditer('\n  "([0-9]+)":\{"tags":\[\n',tail)]
        if (len(allout)==0):
            return (0, frames)
        else:
            return (int(allout[-1].group(1))+1, frames)
        
    def local_execute(self, cmd):
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='') # process line here
    def ls(self):
        print(self.local_dir)
        print(self.local_execute(f'ls -la {self.local_dir}'))
        print(self.remote_dir)
        print(self.config.server.execute(f'ls -la {self.remote_dir}'))
        
    def cat(self, file, remote=True):
        if (remote):
            fullfile=self.instance.remote_experiment_path(file)
            print(f'=== {fullfile}')
            print(self.config.server.execute(f'cat "{fullfile}"'))
        else:
            fullfile=self.instance.local_experiment_path(file)
            print(f'=== {fullfile}')
            self.local_execute(f'cat "{fullfile}"')

    def head(self, file, lines=20, remote=True):
        if (remote):
            fullfile=self.instance.remote_experiment_path(file)
            print(f'=== {fullfile}')
            print(self.config.server.execute(f'head {-lines} "{fullfile}"'))
        else:
            fullfile=self.instance.local_experiment_path(file)
            print(f'=== {fullfile}')
            self.local_execute(f'head {-lines} "{fullfile}"')            
            
    def tail(self, file, lines=20, remote=True):
        if (remote):
            fullfile=self.instance.remote_experiment_path(file)
            print(f'=== {fullfile}')
            print(self.config.server.execute(f'tail {-lines} "{fullfile}"'))
        else:
            fullfile=self.instance.local_experiment_path(file)
            print(f'=== {fullfile}')
            print(local_execute(f'tail {-lines} "{fullfile}"', export=True))
        
    def print_scripts(self):
        self.cat('_run.sh')
        self.cat('_tasks.txt')
        self.cat('0.in')
    
    def run(self, quiet=False):
        self.put_video(quiet=quiet)
        return self.instance.run(1, self.time, cpus_per_worker=self.cpus_per_task, quiet=True)
    def cancel(self):
        jobid=self.jobid()
        if (jobid is None):
            print(f"scancel aborted: no jobid for {self}")
        else:
            print(f"scancel {jobid}, {self}")
            print(self.config.server.execute(f'scancel {jobid}'))

def print_time():
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    print(current_time)
    
from io import StringIO
def sacct(jobs, server, show=False, jobids=None, starttime=None):
    """
    Get sacct info for jobs
    """
    assert (jobs is None) or (jobids is None), "Either jobs or jobids must be None"
    if (jobs is not None):
        jobids = [id for id in map(lambda j:j.jobid(), jobs) if id is not None]
    if (jobids is not None and not jobids): # Defined but empty
        print('Empty jobids')
        res='JobIDRaw|JobName|State|NodeList|NCPUS|Start|Elapsed|Timelimit'
    else:
        if (jobids):
            jobidopt='-j '+( ','.join(map(str,jobids)) )
        else:
            jobidopt=''
        #command = f"sacct -p -j {self._jobid}"
        if (starttime is not None):
            starttimeopt = f'--starttime "{starttime}"'
        else:
            starttimeopt=''
        command = f"sacct {jobidopt} {starttimeopt} --parsable2 -X --format=jobidraw,jobname,state,nodelist,NCPUs,start,elapsed,timelimit"
        print(command)
        res = server.execute(command, timeout=1000)
    if (show):
        print(res)
    else:
        df = pd.read_csv(StringIO(res), sep="|")
        return df
    
def sacct_time(server, time='now-1days', show=False):
    """
    Get sacct info for jobs
    """
    command = f"sacct -S {time} --parsable2 -X --format=jobidraw,jobname,state,nodelist,NCPUs,start,elapsed,timelimit"
    print(command)
    res = server.execute(command, timeout=1000)
    if (show):
        print(res)
    else:
        df = pd.read_csv(StringIO(res), sep="|")
        return df
    
    
import slurmqueen
import paramiko
import os
from os.path import join
import subprocess
    
def local_execute(cmd, export=True):
    res=''
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            if (export):
                res+=line
            else:
                print(line, end='') # process line here
    return res

def pbar_debounced_callback(pbar):
    def get_current_time_milli():
        return int(round(time.time() * 1000))
    def debounce(callback, throttle_time_limit=1000):
        # https://stackoverflow.com/question/31922875/throttle-a-function-call-in-python
        last_millis = get_current_time_milli()  
        def throttle(*args,**kwargs):
            nonlocal last_millis
            curr_millis = get_current_time_milli()
            if (curr_millis - last_millis) > throttle_time_limit:
                last_millis = get_current_time_milli()
                callback(*args,**kwargs)
        return throttle    
    def update_to(current, total):
        pbar.total = total
        pbar.n = current
        pbar.refresh()
    return debounce(update_to)

def check_file(server, local_file, remote_file):
    def _stat(ftp,remote_file):
        try:
            remote_attr = ftp.stat(remote_file)
        except IOError:
            return None
        return remote_attr
    
    local_attr=os.stat(local_file)
    if (local_attr is None):
        return 'NO LOCAL'
    local_size=local_attr.st_size
    with server.ftp_connect() as ftp:
        remote_attr = _stat(ftp,remote_file)
        if (remote_attr is None):
            return 'NO REMOTE'
        remote_size = remote_attr.st_size
        if (remote_size != local_size):
            return 'DIFFERENT SIZE'
              
        res=local_execute(f'tail -c 256 {local_file} | md5sum')
        local_md5 = res[:32]  
        res=server.execute(f'tail -c 256 {remote_file} | md5sum')
        remote_md5 = res[:32]   
        if (local_md5 != remote_md5):
            return 'DIFFERENT MD5'
    return 'SAME'
        
def put_file(server, local_file, remote_file, quiet=False, leave=True):
    def _stat(ftp,remote_file):
        try:
            remote_attr = ftp.stat(remote_file)
        except IOError:
            return None
        return remote_attr
    def _mkdirs(server,path):
        server.execute(f'mkdir -p {path}')
    
    local_attr=os.stat(local_file)
    local_size=local_attr.st_size
    with server.ftp_connect() as ftp:
        remote_attr = _stat(ftp,remote_file)
        if (remote_attr is not None):
            remote_size = remote_attr.st_size
            if (remote_size == local_size):
                print('Same size, Skipping.')
                return
        
        remote_dir = os.path.dirname(remote_file)
        _mkdirs(server, remote_dir)
        
        # File does not exist or different size
        if (quiet):
            ftp.put(local_file,remote_file,confirm=True)
        else:
            print(f"Uploading {local_file} => {remote_file}")
            with tqdm(total=local_size,mininterval=0.5,maxinterval=1.0,unit="B",unit_scale=True,leave=leave) as pbar:
                ftp.put(local_file,remote_file, callback=pbar_debounced_callback(pbar))
                pbar.n=pbar.total; pbar.refresh()
    
def SlurmServerWithKey(server, username, auth_sock, agent_pid, keyname):
    # Launch keychain or ssh-agent with the added key on the terminal first
    # export SSH_AUTH_SOCK=/tmp/ssh-xxxxx/agent.yyyy
    # export SSH_AGENT_PID=yyyy

    agent = paramiko.agent.Agent()
    keys = agent.get_keys()
    self.key=keys[0]

    self.server = slurmqueen.SlurmServer('boqueron.hpcf.upr.edu', 'rmegret', key=self.key)
            
class BoqueronJobManager():
    def __init__(self, server):
        self.aviroot = '/mnt/storage/Gurabo/datasets/gurabo10/avi'
        self.mp4root = '/mnt/storage/Gurabo/datasets/gurabo10/mp4'
        self.tagsroot = '/mnt/storage/Gurabo/datasets/gurabo10/tags'
        
        self.remote_mp4root = '/work/rmegret/rmegret/slurmqueentest/mp4cache/'
        
        # ssh_setup
        if (server is None):
            self.ssh_setup()
        else:
            self.server = server
        
        # exp_setup
        self.expconfig = None
        self.exp_setup()
        
        # jobs
        self.jobs = []
        self.jobs_stored = []
        self.jobs_failed = []
    
    def ssh_setup(self):
        # Launch keychain or ssh-agent with the added key on the terminal first
        # SSH_AUTH_SOCK=/tmp/ssh-xxxxx/agent.yyyy; export SSH_AUTH_SOCK; SSH_AGENT_PID=yyyy; export SSH_AGENT_PID;

        #txt=local_execute('eval `keychain --eval --agents ssh id_rsa_boq 2> /dev/null`; echo $SSH_AUTH_SOCK; echo $SSH_AGENT_PID', export=True)
        txt=local_execute('eval `keychain --eval --agents ssh id_rsa_boq 2> /dev/null`; echo $SSH_AUTH_SOCK; echo $SSH_AGENT_PID', export=True)
        tmp=txt.split('\n')
        os.environ['SSH_AUTH_SOCK'] = tmp[0]
        os.environ['SSH_AGENT_PID'] = tmp[1]

        agent = paramiko.agent.Agent()
        keys = agent.get_keys()
        self.key=keys[0]

        self.server = slurmqueen.SlurmServer('boqueron.hpcf.upr.edu', 'rmegret', key=self.key)
        
    def exp_setup(self):
        self.expconfig = slurmqueen.ExperimentConfig(
            server=self.server,
            partition='batch',
            local_directory='/home/rmegret/beeutils/slurmqueentest/',
            remote_directory='/work/rmegret/rmegret/slurmqueentest/')
        
    ### MP4 handling
    
    def upload_mp4_script(self, df, file=None):
        script = 'eval `keychain --eval --agents ssh id_rsa_boq`\n'
        for key,item in df.iterrows():
            local_mp4 = join(self.mp4root,item.mp4file)
            remote_mp4 = join(self.remote_mp4root,item.mp4file)
            server = 'boqueron.hpcf.upr.edu'
            username = 'rmegret'
            keyfile = '/home/rmegret/.ssh/id_rsa_boq'
            cmd = f'rsync -av -e "ssh -i {keyfile}" {local_mp4} {username}@{server}:{remote_mp4}'
            script += cmd+'\n'
        if (file is None):
            return script
        if (file is not None):
            print(f"Saving rsync script to local file {file}")
            print(f"local_mp4root: {self.mp4root}\nserver: {server}\nusername: {username}\nkeyfile: {keyfile}\nremote_mp4root: {self.remote_mp4root}")
            res = input(f"Overwrite local file? [Y/N]: ")
            if res.upper() == "Y":
                with open(file,'w') as fp:
                    fp.write(script)
            else:
                print("User canceled")
    def upload_mp4(self, df):
        for key,item in df.iterrows():
            local_mp4 = join(self.mp4root,item.mp4file)
            remote_mp4 = join(self.remote_mp4root,item.mp4file)
            res = check_file(self.server, local_mp4, remote_mp4)
            put_file(self.server, local_mp4, remote_mp4, leave=False)    
    def ls_mp4cache(self):
        print(self.server.execute(f'find {self.remote_mp4root}'))
        
    def check_mp4(self, df=None, jobs=None):
        if (jobs is not None and df is not None):
            raise ValueError("Only one of jobs and df can be defined")
        if (jobs is None and df is None):
            jobs=self.jobs
        if (jobs is not None):
            for j in jobs:
                #res = j.check_video()   
                local_mp4 = j.local_mp4
                remote_mp4 = j.remote_mp4
                res = check_file(self.server, local_mp4, remote_mp4)
                print(f"{remote_mp4}: {res}")
        if (df is not None):
            for key,item in df.iterrows():
                local_mp4 = join(self.mp4root,item.mp4file)
                remote_mp4 = join(self.remote_mp4root,item.mp4file)
                res = check_file(self.server, local_mp4, remote_mp4)
                print(f"{remote_mp4}: {res}")
        
    def clean_mp4cache(self, only_if_complete=True, dryrun=True, jobs=None):
        if (jobs is None):
            jobs=self.jobs
            
        for j in jobs:
            j.delete_video(only_if_complete=only_if_complete, dryrun=dryrun)            
        
    ### TAGS COMPUTATION AND STORAGE
        
    def run_tags(self, df, time='10:00:00', quiet=True, cpus_per_task=1, nthreads=1):
        for k,item in df.iterrows():
            print_time()
            j = SlurmJobTag(self.expconfig, item, self.mp4root, time=time, cpus_per_task=cpus_per_task, nthreads=nthreads)
            print(f"{item.mp4file} -- {j.name}")
            self.jobs.append(j)
            j.run(quiet=quiet)
            
    def failed(self, jobs=None):
        mdf=self.status(jobs)
        faileddf=mdf[(mdf.hastags=='NoFile')&((mdf.State=='TIMEOUT')|(mdf.State.str.startswith('CANCELLED')))]
        return faileddf
              
    def clean_failed(self, jobids=None):
        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
              
        if (jobids is None):
              jobids = [i for i in range(len(self.jobs)) if self.failed(self.jobs[i])]
              
        for id in jobids:
            self.jobs_failed.append(self.jobs[id])
        delete_multiple_element(self.jobs, jobids)
            
    def job_from_slurm(self, df, slurmid):
        jdf = sacct(None, self.server, jobids=[slurmid])
        if (jdf.shape[0]==0):
            print(f'Could not find slurm task "{slurmid}"')
            return
        else:
            jitem=jdf.iloc[0]
        #print(jitem)
        mp4file = jitem.JobName.split('__')[1].replace('++','/')+'.mp4'
        print("mp4file",mp4file)
        item = df.loc[df.mp4file==mp4file]
        if (item.shape[0]==0):
            print(f'Could not find {mp4file} in df')
            return
        else:
            item=item.iloc[0]
            #print(item)
        j = SlurmJobTag(self.expconfig, item, self.mp4root, time=jitem.Timelimit, cpus_per_task=jitem.NCPUS, nthreads=jitem.NCPUS, name=jitem.JobName)
        j.instance._jobid=slurmid
        return j    
              
    def jobs_from_slurm(self, df, slurmids=None, starttime=None):
        jobs=[]
        jdf = sacct(None, self.server, jobids=slurmids, starttime=starttime)
        if (jdf.shape[0]==0):
            print(f'Could not find slurm tasks')
            return []
        for k,jitem in jdf.iterrows():
            mp4file = jitem.JobName.split('__')[1].replace('++','/')+'.mp4'
            print("mp4file",mp4file)
            item = df.loc[df.mp4file==mp4file]
            if (item.shape[0]==0):
                print(f'Could not find {mp4file} in df')
                return
            else:
                item=item.iloc[0]
                #print(item)
            j = SlurmJobTag(self.expconfig, item, self.mp4root, time=jitem.Timelimit, cpus_per_task=jitem.NCPUS, nthreads=jitem.NCPUS, name=jitem.JobName)
            j.instance._jobid=jitem.JobIDRaw
            jobs.append(j)
        return jobs
            
    def status(self, jobs=None):
        if (jobs is None):
            jobs = self.jobs
        J = jobs
        ids = range(len(J))
        
        assert J, "Empty Job List"
        
        JL = []
        for id in ids:
            j=jobs[id]
            pp=j.progress()
            obj = dict(
                id=id,
                jobid=j.jobid(),
                hastags=j.check_tags(),
                itemid=j.item.name,
                progress=pp[0],
                frames=pp[1],
                mp4file=j.item.mp4file
            )
            JL.append(obj)
            
        jdf = pd.DataFrame(JL)
            
        #jdf = pd.DataFrame({'id':ids,'jobid':map(lambda j:j.jobid(), J),'hastags':map(lambda j:j.check_tags(), J),'itemid':map(lambda j:j.item.name, J)})
        
        idmap={job.jobid():job for job in J}
        #print(jdf)
        adf=sacct(J, self.server)
        mdf=pd.merge(jdf,adf,how='outer',left_on='jobid',right_on='JobIDRaw').drop('jobid',axis=1)
        mdf=mdf.drop('JobName',axis=1)
        return mdf
    
    def sacct_time(self, time='now-1days', **kwargs):
        adf=sacct_time(self.server, time=time, **kwargs)
        return adf
        
    def get_tags(self, quiet=False):
        for j in self.jobs:
            j.get_tags(quiet=quiet)
    
    def store_tags(self, tagsroot, force=False, dryrun=False, quiet=True):
        storedjobs=[]
        for j in tqdm(self.jobs):
            j.store_tags(tagsroot, force=force, dryrun=dryrun, quiet=quiet)
            if (not dryrun):
                if (j.stored(tagsroot)):
                    storedjobs.append(j)
        if (not dryrun):
            self.jobs_stored = self.jobs_stored+storedjobs
            self.jobs = [j for j in self.jobs if j not in storedjobs]
            print(f"Moved {len(storedjobs)} jobs to jobs_stored")
    
            
                