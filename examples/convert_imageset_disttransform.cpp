// Copyright 2014 Julien Martel
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels which are vectors,
// for instance coming from a distance transform in the format as
//   subfolder1/file1.JPEG 2 3 7 8 5 6 7 
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.
// You are responsible for shuffling the files yourself.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::pair;
using std::string;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4) {
    printf("Convert a set of images to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset ROOTFOLDER/ LISTFILE NUM_LABELS DB_NAME"
        " RANDOM_SHUFFLE_DATA[0 or 1]\n");
    return 0;
  }
  //Arguments to our program
  std::ifstream infile(argv[2]);
  int numLabels = atoi(argv[3]);
  
  // Each line is constituted of the path to the file and the vector of 
  // labels
  std::vector<std::pair<string, std::vector<float> > > lines;
  // --------
  string filename;
  std::vector<float> labels(numLabels);
  
  while (infile >> filename) {
	  for(int l=0; l<numLabels; l++)
		  infile >> (labels[l]);
    lines.push_back(std::make_pair(filename, labels));
    /*
    LOG(ERROR) <<  "filepath: " << lines[lines.size()-1].first;
    LOG(ERROR) << "values: " << lines[lines.size()-1].second[0] 
			   << ","		 << lines[lines.size()-1].second[5] 
			   << ","		 << lines[lines.size()-1].second[8];
			   * */
  }
  if (argc == 5 && argv[5][0] == '1') {
    // randomly shuffle data
    LOG(ERROR) << "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(ERROR) << "A total of " << lines.size() << " images.";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(ERROR) << "Opening leveldb " << argv[4];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[4], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[4];
 
  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int maxKeyLength = 256;
  char key_cstr[maxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    if (!ReadImageWithLabelVectorToDatum(root_folder + lines[line_id].first, lines[line_id].second,
										 &datum)) {
      continue;
    };
  
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
    } else {
      const string& data = datum.data();
      CHECK_EQ(data.size(), data_size) << "Incorrect data field size " << data.size();
    }
    
    // sequential
    snprintf(key_cstr, maxKeyLength, "%08d_%s", line_id, lines[line_id].first.c_str());
    string value;
    
    // get the value
    datum.SerializeToString(&value);
    batch->Put(string(key_cstr), value);
    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR) << "Processed " << count << " files.";
  }

  delete batch;
  delete db;
  return 0;
}

