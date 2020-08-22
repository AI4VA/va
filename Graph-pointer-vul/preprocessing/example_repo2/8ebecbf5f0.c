#include <stdlib.h>           // Tag.OTHER
int main()                    // Tag.OTHER
{                             // Tag.OTHER
    char entity_3[69];        // Tag.BODY
    int entity_2;             // Tag.BODY
    int entity_6;             // Tag.BODY
    entity_2 = rand();        // Tag.BODY
    entity_6 = 89;            // Tag.BODY
    if(entity_2 < entity_6){  // Tag.BODY
    } else {                  // Tag.BODY
    entity_2 = 87;            // Tag.BODY
    }                         // Tag.BODY
    int entity_7;             // Tag.BODY
    entity_7 = 84;            // Tag.BODY
    entity_3[entity_2] = 'p'; // Tag.BUFWRITE_COND_UNSAFE
    char entity_4[63];        // Tag.BODY
    entity_4[entity_7] = 'F'; // Tag.BUFWRITE_TAUT_UNSAFE
    return 0;                 // Tag.BODY
}                             // Tag.OTHER