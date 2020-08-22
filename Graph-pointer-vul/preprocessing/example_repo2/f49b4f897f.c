#include <stdlib.h>             // Tag.OTHER
int main()                      // Tag.OTHER
{                               // Tag.OTHER
    int entity_8;               // Tag.BODY
    char entity_9[77];          // Tag.BODY
    int entity_1;               // Tag.BODY
    int entity_7;               // Tag.BODY
    entity_1 = 14;              // Tag.BODY
    entity_8 = 47;              // Tag.BODY
    int entity_5;               // Tag.BODY
    entity_7 = 54;              // Tag.BODY
    int entity_4;               // Tag.BODY
    char entity_6[21];          // Tag.BODY
    entity_5 = 87;              // Tag.BODY
    entity_4 = rand();          // Tag.BODY
    entity_9[entity_5] = '4';   // Tag.BUFWRITE_TAUT_UNSAFE
    char entity_3[47];          // Tag.BODY
    if (entity_4 < entity_8){   // Tag.BODY
    entity_3[entity_1] = '2';   // Tag.BUFWRITE_TAUT_SAFE
    } else {                    // Tag.BODY
    entity_4 = 42;              // Tag.BODY
    }                           // Tag.BODY
    while(entity_7 < entity_4){ // Tag.BODY
    entity_7++;                 // Tag.BODY
    }                           // Tag.BODY
    entity_6[entity_7] = 'Q';   // Tag.BUFWRITE_COND_UNSAFE
    return 0;                   // Tag.BODY
}                               // Tag.OTHER