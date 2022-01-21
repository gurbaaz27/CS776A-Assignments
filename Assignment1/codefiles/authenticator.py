#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################

# Copyright (c) 2009 Siddharth Agarwal
# Copyright (c) 2021 Hrishikesh Terdalkar
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

###############################################################################

import re
import sys
import time
import logging
import getpass
import argparse

import gc
import netrc
import atexit
import socket

from http.client import (
    HTTPConnection,
    HTTPSConnection,
    HTTPException,
    BadStatusLine,
)
from urllib.parse import urlparse, urlencode, ParseResult

###############################################################################
# Globals, set right in the beginning

username = None
password = None

###############################################################################

ERROR_RETRY_SECONDS = 5
LOGGED_IN_RETRY_SECONDS = 5
LOGGED_IN_SECONDS = 200

HTTP_ADDRESS = "74.125.236.51"

###############################################################################

logger = logging.getLogger("FirewallLogger")

###############################################################################


class FirewallState:
    Start, LoggedIn, End = range(3)


class LoginState:
    AlreadyLoggedIn, InvalidCredentials, Successful = range(3)


###############################################################################


class FirewallAuthenticator:
    """Fortigate Authentication State Machine"""

    def __init__(self, username, password):
        self.handlers = {
            FirewallState.Start: self.start,
            FirewallState.LoggedIn: self.maintain,
            FirewallState.End: sys.exit,
        }

        self.username = username
        self.password = password

        self.state = FirewallState.Start
        self.sleeptime = 0
        self.handler_args = []

    # ----------------------------------------------------------------------- #

    def transition(self):
        (self.state, self.sleeptime, self.handler_args) = self.handlers[
            self.state
        ](*self.handler_args)

    # ----------------------------------------------------------------------- #
    # Transition Functions

    def start(self):
        """
        State function for the Start state

        Attempt logging in.
        If we're already logged in, we can't do anything much.
        If we're not, we should transition to the not-logged-in state.
        """
        try:
            login_state, data = self.login()
        except (HTTPException, socket.error) as e:
            logger.info(
                f"Exception |{e}| while trying to log in. "
                f"Retrying in {ERROR_RETRY_SECONDS} seconds."
            )
            return (FirewallState.Start, ERROR_RETRY_SECONDS, [])

        # Check whether login was successful
        if login_state == LoginState.AlreadyLoggedIn:
            logger.info(
                f"Already logged in (response code {data}). "
                f"Retrying in {LOGGED_IN_RETRY_SECONDS} seconds."
            )
            return (FirewallState.Start, LOGGED_IN_RETRY_SECONDS, [])
        elif login_state == LoginState.InvalidCredentials:
            # Not much we can do.
            return (FirewallState.End, 0, [3])
        else:
            logger.info("Logged in.")
            return (FirewallState.LoggedIn, 0, [data])

    def maintain(self, keepalive_url):
        """
        State function for the LoggedIn state

        Keep the authentication alive by pinging a keepalive URL repeatedly.
        If there are any connection problems, keep trying with the same URL.
        If the keepalive URL doesn't work any more, go back to the start state.
        """

        try:
            self.keep_alive(keepalive_url)
        except BadStatusLine:
            logger.info(
                f"Keepalive URL {keepalive_url.geturl()} doesn't work. "
                "Attempting to log in again."
            )
            return (FirewallState.Start, 0, [])
        except (HTTPException, socket.error) as e:
            logger.info(
                f"Exception |{e}| while trying to keep alive. "
                f"Retrying in {ERROR_RETRY_SECONDS} seconds."
            )
            return (
                FirewallState.LoggedIn,
                ERROR_RETRY_SECONDS,
                [keepalive_url],
            )

        # OK, the URL worked. That's good.
        logger.info("Keeping alive.")
        return (FirewallState.LoggedIn, LOGGED_IN_SECONDS, [keepalive_url])

    # ----------------------------------------------------------------------- #

    def run_forever(self):
        """Run the state machine forever"""

        def atexit_logout():
            """
            Log out from firewall authentication.
            This is supposed to run whenever the program exits.
            """

            if self.state == FirewallState.LoggedIn:
                url = self.handler_args[0]
                logout_url = ParseResult(
                    url.scheme,
                    url.netloc,
                    "/logout",
                    url.params,
                    url.query,
                    url.fragment,
                )
                try:
                    logger.info(f"Logging out with URL {logout_url.geturl()}")
                    conn = HTTPSConnection(logout_url.netloc)
                    conn.request(
                        "GET", f"{logout_url.path}?{logout_url.query}"
                    )
                    response = conn.getresponse()
                    response.read()
                except (HTTPException, socket.error) as e:
                    # Just print an error message
                    logger.warning(f"Exception |{e}| while logging out.")
                finally:
                    conn.close()

        atexit.register(atexit_logout)

        while True:
            self.transition()
            if self.sleeptime > 0:
                time.sleep(self.sleeptime)

    # ----------------------------------------------------------------------- #

    def login(self):
        """
        Attempt to Log In

        Returns
        -------
            AlreadyLoggedIn: If we're already logged in
            InvalidCredentials: If the username/password given are incorrect
            Successful: If we have managed to log in.

        Throws an exception if an error occurs somewhere along the process.
        """
        # Obtain auth url by pinging an HTTP location
        try:
            conn = HTTPConnection(f"{HTTP_ADDRESS}:80")
            conn.request("GET", "/")
            response = conn.getresponse()
            # 303 leads to the auth page, so it means we're not logged in
            if response.status != 303:
                return (LoginState.AlreadyLoggedIn, response.status)

            authlocation = response.getheader("Location")
        finally:
            conn.close()

        logger.info(f"The auth location is: {authlocation}")

        # Make a connection to the auth location
        parsedauthloc = urlparse(authlocation)
        try:
            logging.debug(parsedauthloc.netloc)
            authconn = HTTPSConnection(parsedauthloc.netloc)
            authconn.request(
                "GET", parsedauthloc.path + "?" + parsedauthloc.query
            )
            response = authconn.getresponse()
            data = response.read().decode("utf-8")
        finally:
            authconn.close()

        # Look for the right magic value in the data
        match = re.search(r"VALUE=\"([0-9a-f]+)\"", data, re.IGNORECASE)
        magicString = match.group(1)
        logger.debug("The magic string is: " + magicString)

        # Now construct a POST request
        params = urlencode(
            {
                "username": self.username,
                "password": self.password,
                "magic": magicString,
                "4Tredir": "/",
            }
        )
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
        }

        try:
            postconn = HTTPSConnection(parsedauthloc.netloc)
            postconn.request("POST", "/", params, headers)

            # Get the response
            post_response = postconn.getresponse()
            post_data = post_response.read().decode("utf-8")
        finally:
            postconn.close()

        # Look for the keepalive URL
        keepalive_match = re.search(r"a.href=\"(.+?)\"", post_data)
        if keepalive_match is None:
            # Whoops, unsuccessful
            # Probably the username and password didn't match
            logger.fatal("Authentication failed. Are the credentials correct?")
            return (LoginState.InvalidCredentials, None)

        keepalive_url = keepalive_match.group(1)

        logger.info(f"The keep alive URL is: {keepalive_url}")
        logger.debug(post_data)
        return (LoginState.Successful, urlparse(keepalive_url))

    @staticmethod
    def keep_alive(url):
        """Keep the connection alive by pinging a URL"""
        logger.info("Attempting to keep alive")
        try:
            conn = HTTPSConnection(url.netloc)
            conn.request("GET", f"{url.path}?{url.query}")
            # This line raises an exception if the URL stops working.
            # We catch it in logged_in_func.
            response = conn.getresponse()

            logger.debug(str(response.status))
            logger.debug(response.read().decode('utf-8'))
        finally:
            conn.close()
            gc.collect()

    # ----------------------------------------------------------------------- #


###############################################################################
# Utility Functions


def get_credentials(username=None, password=None, use_netrc=None):
    """
    Get the username and password

    Fetches the credentials from netrc if use_netrc is True.
    Fetches the missing credentials interactively.
    """
    if use_netrc:
        try:
            info = netrc.netrc()
            cred = info.authenticators("172.31.1.251")
            if cred:
                return (cred[0], cred[2])
            logger.info("Could not find credentials in netrc file.")
        except Exception:
            logger.info("Could not read from netrc file.")

    if username is None:
        # Get the username from the input
        username = input("Username: ")

    if password is None:
        # Read the password without echoing it
        password = getpass.getpass()

    return (username, password)


def setup_logger(logger_name=None, verbose=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    if verbose:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Fortigate Authenticator")
    parser.add_argument("-u", "--username", help="Username")
    parser.add_argument("-p" "--password", help="Password")
    parser.add_argument(
        "-n",
        "--netrc",
        action="store_true",
        dest="netrc",
        help="Read credentials from netrc file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print debugging information",
    )

    args = vars(parser.parse_args())

    # Setup Logger
    setup_logger(verbose=args["verbose"])

    # Try authenticating!
    global username, password
    username, password = get_credentials(
        username=args.get("username"),
        password=args.get("password"),
        use_netrc=args.get("netrc"),
    )

    authenticator = FirewallAuthenticator(username=username, password=password)
    authenticator.run_forever()

    return 0


###############################################################################

if __name__ == "__main__":
    sys.exit(main())