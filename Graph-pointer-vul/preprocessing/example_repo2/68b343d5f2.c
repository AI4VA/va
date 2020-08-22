#include <stdlib.h>             // Tag.OTHER
int main()                      // Tag.OTHER
{                               // Tag.OTHER
    char entity_7[24];          // Tag.BODY
    int entity_9;               // Tag.BODY
    int entity_8;               // Tag.BODY
    entity_8 = 61;              // Tag.BODY
    int entity_4;               // Tag.BODY
    char entity_6[70];          // Tag.BODY
    entity_4 = 72;              // Tag.BODY
    entity_9 = 56;              // Tag.BODY
    entity_7[entity_4] = 'e';   // Tag.BUFWRITE_TAUT_UNSAFE
    while(entity_8 < entity_9){ // Tag.BODY
    entity_8++;                 // Tag.BODY
    }                           // Tag.BODY
    int entity_5;               // Tag.BODY
    entity_6[entity_8] = 'T';   // Tag.BUFWRITE_COND_SAFE
    entity_5 = 80;              // Tag.BODY
    char entity_1[85];          // Tag.BODY
    entity_1[entity_5] = '0';   // Tag.BUFWRITE_TAUT_SAFE
    return 0;                   // Tag.BODY
}                               // Tag.OTHER