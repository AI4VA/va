#include <stdlib.h>                                     // Tag.OTHER
int main()                                              // Tag.OTHER
{                                                       // Tag.OTHER
    int entity_2;                                       // Tag.BODY
    int entity_8;                                       // Tag.BODY
    char entity_1[8];                                   // Tag.BODY
    entity_8 = 33;                                      // Tag.BODY
    int entity_6;                                       // Tag.BODY
    entity_2 = rand();                                  // Tag.BODY
    if (entity_2 < entity_8){                           // Tag.BODY
    } else {                                            // Tag.BODY
    entity_2 = 28;                                      // Tag.BODY
    }                                                   // Tag.BODY
    for(entity_6 = 9; entity_6 < entity_2; entity_6++){ // Tag.BODY
    }                                                   // Tag.BODY
    int entity_3;                                       // Tag.BODY
    entity_1[entity_6] = 'F';                           // Tag.BUFWRITE_COND_UNSAFE
    entity_3 = 76;                                      // Tag.BODY
    char entity_7[5];                                   // Tag.BODY
    entity_7[entity_3] = '6';                           // Tag.BUFWRITE_TAUT_UNSAFE
    return 0;                                           // Tag.BODY
}                                                       // Tag.OTHER