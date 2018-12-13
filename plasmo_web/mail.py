# -*- coding: utf-8 -*-
"""
This file implements what is needed to send emails from the application.

Adapted by Laura Duciel
Contact : laura.duciel@casc4de.eu
"""

from __future__ import print_function
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import os.path
###

class GMAIL(object):
    '''
    Class for sending mails with Gmail with smtp protocol.
    input:
        gmail_user : name of the Gmail account
        gmail_pwd  : password of the Gmail account
        to: destination of the mail
        subject: subject of the mail
        text: text to be sent
        attach: List of Attached documents
    Usage: 
        import Sendmail from this file and call SendMail(to,subject,text,attach)
    '''
    def __init__(self, gmail_user = "anything@gmail.com", gmail_pwd = "anything" ):
        self.gmail_user = gmail_user
        self.gmail_pwd = gmail_pwd

    def send(self, to, cc="", bcc="", subject="", text = "", attach = None):
        self.to = to
        self.cc = cc
        self.bcc = bcc
        self.subject = subject
        self.text = text
        self.attach = attach
        msg = MIMEMultipart()
        #################
        msg['From'] = self.gmail_user
        msg['To'] = self.to
        if self.cc != "":
            msg['Cc'] = self.cc
        msg['Subject'] = self.subject
        #############
        if text != '' :
           msg.attach(MIMEText(self.text, _charset="utf-8"))
        ###############
        if attach is not None:
            if isinstance(self.attach, str) : # attach is either a filename
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(open(attach, 'rb').read())
                Encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                    'attachment; filename = "%s"' % os.path.basename(attach))
                msg.attach(part)
            else:                       # or an iterable of filenames
                for att in self.attach:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(open(att, 'rb').read())
                    Encoders.encode_base64(part)
                    part.add_header('Content-Disposition',
                        'attachment; filename = "%s"' % os.path.basename(att))
                    msg.attach(part)
        ##############
        mailServer = smtplib.SMTP("smtp.gmail.com", 587)
        mailServer.ehlo()
        mailServer.starttls()
        mailServer.ehlo()
        mailServer.login(self.gmail_user, self.gmail_pwd)
        rcpt = self.cc.split(",") + self.bcc.split(",") + self.to.split(",")
        mailServer.sendmail(self.gmail_user, rcpt, msg.as_string())
        mailServer.close()
        
class SendMail():
    def __init__(self, to= 'testflaskmail@gmail.com', cc="", bcc="", subject='automatic mail', text="hello", attach=None):
        gm = GMAIL()
        gm.send(to=to, cc=cc, bcc=bcc, subject=subject, text=text, attach=attach)
if __name__ == '__main__':
    # TEST
    SendMail(to="madelsuc@unistra.fr, mad@delsuc.net",
        bcc="delsuc@igbmc.fr",
        subject="Testing Sendmail script - list+bcc2",
        text="Text of test",
        attach=["mail.py"])
    


