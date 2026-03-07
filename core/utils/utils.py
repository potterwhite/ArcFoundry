# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import select


def timed_input(prompt, timeout=30, default='y'):
    """
    Waits for user input with a countdown. Returns default if timeout.
    Works on Linux/Unix systems.
    """
    sys.stdout.write(
        f"{prompt} (Timeout: {timeout}s, Default: {default.upper()}): ")
    sys.stdout.flush()

    # Monitor sys.stdin for input
    ready, _, _ = select.select([sys.stdin], [], [], timeout)

    if ready:
        user_input = sys.stdin.readline().strip().lower()
        return user_input if user_input else default
    else:
        sys.stdout.write(
            f"\n[TIMEOUT] No input received. Auto-selecting default: {default.upper()}\n"
        )
        return default
