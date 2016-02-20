import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import aws_s3_utility

class TestAWSS3Utility(unittest.TestCase):

    def test_directory_upload(self):
        ak = ''
        sk = ''
        bucket = 'hierarchical'
        aws_util = aws_s3_utility.S3Utility(ak, sk, bucket)
        directory = '.'
        aws_util.upload_directory('/Users/wulfe/Desktop/aws_run/NeuralAgent_2016-02-20T02:59:50.808542')

if __name__ == '__main__':
    unittest.main()

