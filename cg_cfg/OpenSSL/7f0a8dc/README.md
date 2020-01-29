### Call Graph / Control Flow Graph Examples from OpenSSL

#### OpenSSL version:

committ `7f0a8dc7f9c5c35af0f66aca553304737931d55f`, the version we manually studied.

```shell
git clone https://github.com/openssl/openssl.git

git checkout 7f0a8dc7f9c5c35af0f66aca553304737931d55f
```

#### Call Graphs

OpenSSL generates multiple binaries. We dump one call graph per binary. The list of binaries could be found in `exec_file_list.txt`. The corresponding call graphs are in `cg`.


The call graphs are in dot format. an example
```
digraph "Call graph" {
        label="Call graph";
        ......
        Node0x55c50be7d5f0 [shape=record,label="{main}"];
        Node0x55c50be7d5f0 -> Node0x55c50be7d660;
        Node0x55c50be7d5f0 -> Node0x55c50bfcb840;
        Node0x55c50be7d660 [shape=record,label="{make_config_name}"];
        Node0x55c50bfcb840 [shape=record,label="{dup_bio_in}"];
        ......
        
```

the call edges are represetned by `func_ID_1 -> func_ID_2;`, where the names for `func_ID_` can be found in the `label` field in `func_ID_ [shape=record,label="{function name}"];`


#### Control Flow Graphs
TBD
