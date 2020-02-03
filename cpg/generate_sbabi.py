import os
import shutil
import json

#tags 3 or 5 == vuln
#tags 2 or 4 == non_vuln

def read_manifest(filename):
    fd = open(filename)
    return json.load(fd)

def count_classes(manifest):
    ctr_total = 0
    ctr_only_vuln = 0
    ctr_vuln = 0
    tags_dict = manifest['tags']
    for _file in tags_dict:
        ctr_total += 1
        tags = tags_dict[_file]
        if 3 in tags or 5 in tags:
            ctr_vuln += 1
            if 2 not in tags and 4 not in tags:
                ctr_only_vuln += 1
    print(ctr_only_vuln, ctr_vuln, ctr_total)        
    

# works for --linear_only and --taut_only
def move_file1(filename, tags, src_dir, dst_dir_vuln, dst_dir_nonvuln):
    src_path = src_dir + '/' + filename
    dst_path = None
    if 3 in tags or 5 in tags:
        if 2 not in tags and 4 not in tags:
            dst_path = dst_dir_vuln + '/' + filename
    else:
        if 2 in tags or 4 in tags:
            dst_path = dst_dir_nonvuln + '/' + filename
    if dst_path:
        shutil.move(src_path, dst_path)


def move_file(filename, tags, src_dir, dst_dir_vuln, dst_dir_nonvuln):
    src_path = src_dir + '/' + filename
    dst_path = None
    if 3 in tags and 5 not in tags: #cond_unsafe
        if 2 not in tags and 4 not in tags:
            dst_path = dst_dir_vuln + '/' + filename
    if 2 in tags and 4 not in tags: #cond_safe
        if 3 not in tags and 5 not in tags:
            dst_path = dst_dir_nonvuln + '/' + filename
    if dst_path:
        shutil.move(src_path, dst_path)



def balance_files(manifest,src_dir):
    dir_vuln = src_dir + '/vuln'
    dir_nonvuln = src_dir + '/nonvuln'
    assert(os.path.exists(dir_vuln))
    assert(os.path.exists(dir_nonvuln))

    vuln_files = os.listdir(dir_vuln)
    nonvuln_files = os.listdir(dir_nonvuln)
    num_vuln_files = len(vuln_files)
    num_nonvuln_files = len(nonvuln_files)
    
    if num_vuln_files > num_nonvuln_files:
        print("more vuln files; balancing...")
        _dir = src_dir + '/vuln'
        _dir_extra = src_dir + '/vuln_extra'
        files = vuln_files
        target_numfiles = num_nonvuln_files
    else:    
        print("more nonvuln files; balancing...")
        _dir = src_dir + '/nonvuln'
        _dir_extra = src_dir + '/nonvuln_extra'
        files = nonvuln_files
        target_numfiles = num_vuln_files

    assert(not os.path.exists(_dir_extra))
    os.mkdir(_dir_extra)

    ctr = 0
    ctr_moved = 9
    for _file in files:
        ctr += 1
        if ctr > target_numfiles:
            src_path = _dir + '/' + _file
            dst_path = _dir_extra + '/' + _file
            shutil.move(src_path, dst_path)
            ctr_moved += 1
    print("num files moved: ", ctr_moved)



def separate_files(manifest,src_dir):
    dst_dir_vuln = src_dir + '/vuln'
    dst_dir_nonvuln = src_dir + '/nonvuln'
    os.mkdir(dst_dir_vuln)
    os.mkdir(dst_dir_nonvuln)

    tags_dict = manifest['tags']
    for _file in tags_dict:
        tags = tags_dict[_file]
        move_file(_file, tags, src_dir, dst_dir_vuln, dst_dir_nonvuln)

def create_files():
    # python3.6 sa_babi/generate.py  -seed 0 -num_instances 60000 -metadata_file /mnt/m1/sbabi/sa-bAbI/data/test/manifest.json  /mnt/m1/sbabi/sa-bAbI/data/test
    # python3.6 sa_babi/generate.py  -seed 1 -num_instances 60000 -metadata_file /mnt/m1/sbabi/sa-bAbI/data/valid/manifest.json  /mnt/m1/sbabi/sa-bAbI/data/valid
    # python3.6 sa_babi/generate.py  -seed 2 -num_instances 600000 -metadata_file /mnt/m1/sbabi/sa-bAbI/data/train/manifest.json  /mnt/m1/sbabi/sa-bAbI/data/train
    # config options --linear_only, --taut_only
    pass

src_dir = '/mnt/m1/sbabi/sa-bAbI/data/train'
manifest_file = src_dir + '/manifest.json'
manifest = read_manifest(manifest_file)
count_classes(manifest)
separate_files(manifest, src_dir)
balance_files(manifest, src_dir)
