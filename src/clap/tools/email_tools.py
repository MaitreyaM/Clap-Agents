
import os
import smtplib
import imaplib
import email
import json 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
from dotenv import load_dotenv
import anyio 
import functools 
import requests
from typing import Optional

from clap.tool_pattern.tool import tool

load_dotenv()

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
IMAP_HOST = "imap.gmail.com"
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


def _send_email_sync(recipient: str, subject: str, body: str, attachment_path: Optional[str] = None) -> str:
    """Synchronous helper to send email."""
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return "Error: SMTP username or password not configured in environment."
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, recipient, msg.as_string())
        
        if attachment_path and attachment_path.startswith("temp_attachments"):
            try: os.remove(attachment_path)
            except OSError: pass 
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"

def _download_attachment_sync(attachment_url: str, attachment_filename: str) -> str:
    """Synchronous helper to download an attachment."""
    temp_dir = "temp_attachments" 
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, attachment_filename)
    
    with requests.get(attachment_url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return file_path

def _get_pre_staged_attachment_sync(attachment_name: str) -> Optional[str]:
    """Synchronous helper to get a pre-staged attachment."""
    attachment_dir = "available_attachments" 
    file_path = os.path.join(attachment_dir, attachment_name)
    return file_path if os.path.exists(file_path) else None

def _fetch_emails_sync(folder: str, limit: int) -> str:
    """Synchronous helper to fetch emails."""
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return "Error: Email username or password not configured in environment."
    emails_data = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST)
        mail.login(SMTP_USERNAME, SMTP_PASSWORD)
        status, messages = mail.select(folder)
        if status != 'OK':
            mail.logout()
            return f"Error selecting folder '{folder}': {messages}"

        result, data = mail.search(None, "ALL")
        if status != 'OK' or not data or not data[0]:
            mail.logout()
            return f"No emails found in folder '{folder}'."

        email_ids = data[0].split()
        
        ids_to_fetch = email_ids[-(limit):]

        for email_id_bytes in reversed(ids_to_fetch):
            status, msg_data = mail.fetch(email_id_bytes, "(RFC822)")
            if status == 'OK':
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or "utf-8")
                        from_ = msg.get("From", "")
                        date_ = msg.get("Date", "")
                        
                        snippet = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                ctype = part.get_content_type()
                                cdisp = str(part.get("Content-Disposition"))
                                if ctype == "text/plain" and "attachment" not in cdisp:
                                    try:
                                        body = part.get_payload(decode=True)
                                        snippet = body.decode(part.get_content_charset() or 'utf-8')
                                        snippet = " ".join(snippet.splitlines()) 
                                        snippet = snippet[:150] + "..." 
                                        break
                                    except Exception:
                                        snippet = "[Could not decode body]"
                                        break
                        else:
                            try:
                                body = msg.get_payload(decode=True)
                                snippet = body.decode(msg.get_content_charset() or 'utf-8')
                                snippet = " ".join(snippet.splitlines())
                                snippet = snippet[:150] + "..."
                            except Exception:
                                snippet = "[Could not decode body]"

                        emails_data.append({
                            "id": email_id_bytes.decode(),
                            "from": from_,
                            "subject": subject,
                            "date": date_,
                            "snippet": snippet
                        })
            if len(emails_data) >= limit: 
                 break

        mail.logout()

        if not emails_data:
            return f"No emails found in folder '{folder}'."

        
        result_text = f"Recent emails from {folder} (up to {limit}):\n\n"
        for i, email_data in enumerate(emails_data, 1):
            result_text += f"{i}. From: {email_data['from']}\n"
            result_text += f"   Subject: {email_data['subject']}\n"
            result_text += f"   Date: {email_data['date']}\n"
            result_text += f"   Snippet: {email_data['snippet']}\n\n"
            # result_text += f"   ID: {email_data['id']}\n\n" # ID might not be useful for LLM
        return result_text.strip()
    except Exception as e:
        return f"Failed to fetch emails: {e}"



@tool
async def send_email(recipient: str, subject: str, body: str,
                     attachment_path: Optional[str] = None,
                     attachment_url: Optional[str] = None,
                     attachment_name: Optional[str] = None) -> str:
    """
    Sends an email using configured Gmail account. Can handle attachments via local path, URL, or pre-staged name.

    Args:
        recipient: The email address to send the email to.
        subject: The email subject.
        body: The email body text.
        attachment_path: Optional direct file path for an attachment.
        attachment_url: Optional URL from which to download an attachment (requires attachment_name).
        attachment_name: Optional filename for the attachment (used with URL or pre-staged).

    Returns:
        Success or error message string.
    """
    final_attachment_path = attachment_path
    if attachment_url and attachment_name:
        try:
            # Run synchronous download in thread
            print(f"[Email Tool] Downloading attachment from {attachment_url}...")
            final_attachment_path = await anyio.to_thread.run_sync(
                _download_attachment_sync, attachment_url, attachment_name
            )
            print(f"[Email Tool] Attachment downloaded to {final_attachment_path}")
        except Exception as e:
            return f"Failed to download attachment from URL: {e}"
    elif attachment_name:
        try:
            
            print(f"[Email Tool] Checking for pre-staged attachment: {attachment_name}...")
            final_attachment_path = await anyio.to_thread.run_sync(
                _get_pre_staged_attachment_sync, attachment_name
            )
            if not final_attachment_path:
                return f"Error: Attachment '{attachment_name}' not found in pre-staged directory 'available_attachments'."
            print(f"[Email Tool] Using pre-staged attachment: {final_attachment_path}")
        except Exception as e:
             return f"Error accessing pre-staged attachment: {e}"

    
    print(f"[Email Tool] Sending email to {recipient}...")
    return await anyio.to_thread.run_sync(
        _send_email_sync, recipient, subject, body, final_attachment_path
    )

@tool
async def fetch_recent_emails(folder: str = "INBOX", limit: int = 5) -> str:
    """
    Fetches subject, sender, date, and a snippet of recent emails (up to limit) from a specified folder.

    Args:
        folder: The email folder to fetch from (default: "INBOX"). Common options: "INBOX", "Sent", "Drafts", "[Gmail]/Spam", "[Gmail]/Trash".
        limit: Maximum number of emails to fetch (default: 5).

    Returns:
        A formatted string containing details of the recent emails or an error message.
    """
    print(f"[Email Tool] Fetching up to {limit} emails from folder '{folder}'...")
    return await anyio.to_thread.run_sync(_fetch_emails_sync, folder, limit)
