import os
import sys
import boto

from boto.s3.key import Key
from boto.s3.connection import S3Connection

class S3Utility(object):
    """
    :description: An AWS S3 utility class.

    This is a class rather than a module because some state is required 
    to avoid having to reestablish an s3 connection each transaction.

    :type access_key: string
    :param access_key: aws access key

    :type secret_key: string
    :param secret_key: aws secret key

    """

    def __init__(self, access_key, secret_key, s3_bucket):
        self.access_key = access_key
        self.secret_key = secret_key
        self.s3_bucket = s3_bucket
        self._conn = None

    @property   
    def conn(self):
        if self._conn is not None:
            return self._conn
        else:
            return S3Connection(self.access_key, self.secret_key)

    def download_file_list(self, prefix=''):
        """
        :description: loads the name of the files in a bucket. 
            Optionally returns only those filenames that start with prefix.
        """

        # select the bucket, where input_s3_bucket takes the form 'bsdsdata'
        bucket = self.conn.get_bucket(self.s3_bucket)

        # collect the list of files to process - those that start with the data group id
        file_list = []
        for key in bucket.list():
            key_name = key.name.encode('utf-8')
            if key_name.startswith(prefix):
                file_list.append(key_name)

        return file_list

    def download_file(self, file_to_load, local_save_dir):
        """
        :description: load a file from a given s3 bucket with a 
            given name and save to a local dir

        :type s3_bucket: string
        :param s3_bucket: s3 bucket from which to load the file

        :type file_to_load: string
        :param file_to_load: the file to load

        :type local_save_dir: string
        :param local_save_dir: the local dir to which to save the downloaded file

        :return: the location where the file was saved
        """

        # select the bucket, where input_s3_bucket takes the form 'bsdsdata'
        bucket = self.conn.get_bucket(self.s3_bucket)

        # set a key to the processed files list
        key = Key(bucket, file_to_load)
        key_name = key.name.encode('utf-8')

        # download the file to process and save in the input location
        save_location = os.path.join(local_save_dir, key_name)
        try:
            key.get_contents_to_filename(save_location)
        except boto.exception.S3ResponseError as e:
            raise boto.exception.S3ResponseError("key name: {} failed".format(key_name))

        # return the location of the downloaded file
        return save_location

    def upload_file(self, filename_to_save_as, file_path):
        """
        :description: uploads a single file to an s3 bucket
        """
        # what is this?
        def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()

        # select the bucket, where input_s3_bucket takes the form 'bsdsdata'
        bucket = self.conn.get_bucket(self.s3_bucket)

        # send the file to the s3 bucket
        key = Key(bucket)
        key.key = filename_to_save_as
        key.set_contents_from_filename(file_path, cb=percent_cb, num_cb=50)

    def upload_directory(self, directory):
        """
        :description: upload all the files in a directory to aws s3
        """

        filepaths = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepaths.append(os.path.join(root, filename))

        upload_directory = os.path.basename(directory)
        for filepath in filepaths:
            dest_filepath = os.path.join(upload_directory, filepath.split(upload_directory)[-1][1:])
            self.upload_file(dest_filepath, filepath)


