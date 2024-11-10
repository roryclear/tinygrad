#import "ViewController.h"
#import <sys/socket.h>
#import <netinet/in.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <GZIP/GZIP.h>

@interface ViewController ()
@property (nonatomic) CFSocketRef socket;
@end

@implementation ViewController

NSMutableDictionary<NSString *, id> *objects;
id<MTLComputeCommandEncoder> encoder;
id<MTLDevice> device;
id<MTLCommandQueue> command_queue;
MTLComputePipelineDescriptor *desc;
NSMutableArray *queue;

- (void)viewDidLoad {
    objects = [[NSMutableDictionary alloc] init];
    device = MTLCreateSystemDefaultDevice();
    [objects setObject: device forKey:@"d"]; //DEVICE
    queue = [[NSMutableArray alloc] init];
    [super viewDidLoad];
    [self startHTTPServer];
}

uint8_t *convertNSStringToBytes(NSString *hexString) {
    uint8_t *bytes = malloc(4 * sizeof(uint8_t));
    NSArray<NSString *> *components = [hexString componentsSeparatedByString:@" "];
    for (NSInteger i = 0; i < components.count; i++) {
        unsigned int byteValue;
        [[NSScanner scannerWithString:components[i]] scanHexInt:&byteValue];
        bytes[i] = (uint8_t)byteValue;
    }
    return bytes;
}

char *charArrayFromMTLBuffer(id<MTLBuffer> buffer) {
    uint8_t *bytes = (uint8_t *)buffer.contents;
    NSUInteger length = buffer.length;
    char *hexString = malloc(length * 3);
    if (hexString == NULL) return NULL;
    char *p = hexString;
    for (NSUInteger i = 0; i < length; i++) {
        p += sprintf(p, "%02x", bytes[i]);
        if (i < length - 1) {
            *p++ = ' ';
        }
    }
    *p = '\0';
    return hexString;
}

- (void)startHTTPServer {
    self.socket = CFSocketCreate(NULL, PF_INET, SOCK_STREAM, IPPROTO_TCP, kCFSocketAcceptCallBack, AcceptCallback, NULL);
    if (!self.socket) {
        NSLog(@"Unable to create socket.");
        return;
    }
    
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_len = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_port = htons(8081);  //use same port on tinygrad
    address.sin_addr.s_addr = INADDR_ANY;
    
    CFDataRef addressData = CFDataCreate(NULL, (const UInt8 *)&address, sizeof(address));
    if (CFSocketSetAddress(self.socket, addressData) != kCFSocketSuccess) {
        NSLog(@"Failed to bind socket to address.");
        CFRelease(self.socket);
        self.socket = NULL;
        exit(0); //TODO, add ui or retry
        return;
    }
    CFRelease(addressData);
    
    CFRunLoopSourceRef source = CFSocketCreateRunLoopSource(NULL, self.socket, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);
    CFRelease(source);
    
    NSLog(@"HTTP Server started on port 8081.");
}

void printBufferBytes(id<MTLBuffer> buffer) {
    unsigned char *bytes = (unsigned char *)[buffer contents];
    NSUInteger length = [buffer length];
    NSMutableString *byteString = [NSMutableString stringWithCapacity:length * 3];
    for (NSUInteger i = 0; i < length; i++) {
        [byteString appendFormat:@"%02x ", bytes[i]];
    }
    NSLog(@"Buffer bytes: %@", byteString);
}


static void AcceptCallback(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address, const void *data, void *info) {
    if (type != kCFSocketAcceptCallBack) return;
    
    CFSocketNativeHandle handle = *(CFSocketNativeHandle *)data;
    char buffer[1024*500] = {0}; //TODO how big/small should this be?
    
    ssize_t receivedBytes = recv(handle, buffer, sizeof(buffer) - 1, 0);
    if (receivedBytes < 1) {
        NSLog(@"Failed to receive data.");
        close(handle);
        return;
    }

    buffer[receivedBytes] = '\0';
    CFDataRef dataRef = CFDataCreate(NULL, (UInt8 *)buffer, (CFIndex)receivedBytes);
    CFHTTPMessageRef httpRequest = CFHTTPMessageCreateEmpty(NULL, TRUE);
    CFHTTPMessageAppendBytes(httpRequest, CFDataGetBytePtr(dataRef), CFDataGetLength(dataRef));
    
    if (CFHTTPMessageIsHeaderComplete(httpRequest)) {
        //NSData *bodyData = (__bridge_transfer NSData *)CFHTTPMessageCopyBody(httpRequest);
        NSData *bodyDataUnc = (__bridge_transfer NSData *)CFHTTPMessageCopyBody(httpRequest);
        NSData *bodyData = [bodyDataUnc gunzippedData];
        
        if (!bodyData) {
            const char *response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nInvalid request: Missing or malformed body.";
            send(handle, response, strlen(response), 0);
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }

        NSError *jsonError = nil;
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:bodyData options:0 error:&jsonError];
        
        if (!jsonDict || jsonError) {
            const char *response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nInvalid request: Missing or malformed body.";
            send(handle, response, strlen(response), 0);
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }
        
        NSArray *req_queue = jsonDict[@"queue"];
        [queue addObjectsFromArray:req_queue];
        for(int j = 0; j < [queue count]; j++) {
            if([queue[j][1] isEqualToString:@"dispatchThreadgroups:threadsPerThreadgroup:"]){ //TODO, don't know how to not hardcode yet, copies twice atm too
                [objects[queue[j][0]] dispatchThreadgroups: MTLSizeMake([queue[j][3] intValue], [queue[j][4] intValue], [queue[j][5] intValue]) threadsPerThreadgroup: MTLSizeMake([queue[j][6] intValue], [queue[j][7] intValue], [queue[j][8] intValue]) ];
            } else if([queue[j][1] isEqualToString:@"concurrentDispatchThreadgroups:threadsPerThreadgroup:"]){
                [objects[queue[j][0]] concurrentDispatchThreadgroups: MTLSizeMake([queue[j][3] intValue], [queue[j][4] intValue], [queue[j][5] intValue]) threadsPerThreadgroup: MTLSizeMake([queue[j][6] intValue], [queue[j][7] intValue], [queue[j][8] intValue]) ];
            } else if([queue[j][1] isEqualToString:@"useResources:count:usage:"]){
                NSInteger count = [queue[j] count] - 4;
                MTLResourceUsage usage = MTLResourceUsageRead | MTLResourceUsageWrite;
                NSMutableArray<id<MTLResource>> *resources = [NSMutableArray array];
                for(int i = 0; i < count; i++){
                    [resources addObject:objects[queue[j][i+3]]];
                }
                __unsafe_unretained id<MTLResource> resourceArray[resources.count];
                [resources getObjects:resourceArray range:NSMakeRange(0, resources.count)];
                [objects[queue[j][0]] useResources:resourceArray count:count usage:usage];
            } else if([queue[j][0] isEqualToString:@"delete"]) {
                [objects removeAllObjects];
                [objects setObject: device forKey:@"d"]; //NEED TO KEEP DEVICE
            } else if([queue[j][1] isEqualToString:@"executeCommandsInBuffer:withRange:"]) {
                [objects[queue[j][0]] executeCommandsInBuffer:objects[queue[j][3]] withRange:NSMakeRange(0,[queue[j][4] intValue])];
            } else if([queue[j][0] isEqualToString:@"copyin"]) {
                NSArray<NSString *> *hexArray = [queue[j][1] componentsSeparatedByString:@" "];
                NSUInteger length = hexArray.count;
                uint8_t *bytes = malloc(length);
                for (NSUInteger i = 0; i < length; i++) {
                    unsigned int byteValue;
                    [[NSScanner scannerWithString:hexArray[i]] scanHexInt:&byteValue];
                    bytes[i] = (uint8_t)byteValue;
                }
                NSData *data = [NSData dataWithBytesNoCopy:bytes length:length freeWhenDone:YES];
                memcpy([(id<MTLBuffer>)objects[queue[j][3]] contents], [data bytes], [data length]);
            } else if([queue[j][0] isEqualToString:@"copyout"]) {
                char *bytes = charArrayFromMTLBuffer(objects[queue[j][1]]);
                char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
                size_t totalLength = strlen(response) + strlen(bytes) + 1;
                char *fullResponse = malloc(totalLength);
                strcpy(fullResponse, response);
                strcat(fullResponse, bytes);
                send(handle, fullResponse, strlen(fullResponse), 0);
                close(handle);
                [queue removeAllObjects];
                return;
            } else if([queue[j][0] isEqualToString:@"memcpy"]) {
                if ([objects objectForKey:queue[j][3]] == nil) {
                    [objects setObject:[NSData dataWithContentsOfURL:[[NSBundle mainBundle] URLForResource:queue[j][3] withExtension:nil]] forKey:queue[j][3]];
                }
                memcpy([(id<MTLBuffer>)objects[queue[j][1]] contents] + 0, [(NSData *)objects[queue[j][3]] bytes] + [queue[j][4] intValue], [queue[j][5] intValue]);
            } else {
                SEL selector = NSSelectorFromString(queue[j][1]);
                NSMethodSignature *signature;
                NSInvocation *invocation;
                if ([objects objectForKey:queue[j][0]]) {
                    signature = [(id)objects[queue[j][0]] methodSignatureForSelector:selector];
                    invocation = [NSInvocation invocationWithMethodSignature:signature];
                    [invocation setSelector:selector];
                    [invocation setTarget:objects[queue[j][0]]];
                } else {
                    Class class = NSClassFromString(queue[j][0]);
                    NSMethodSignature *signature = [class methodSignatureForSelector:selector];
                    invocation = [NSInvocation invocationWithMethodSignature:signature];
                    [invocation setSelector:selector];
                    [invocation setTarget:class];
                }
                for(int i = 3; i < 3+[queue[j][2] intValue]; i++){
                    if ([queue[j][i] isKindOfClass:[NSNumber class]]) {
                        [invocation setArgument:&(NSInteger){[queue[j][i] intValue]} atIndex:i-1];
                        continue;
                    }
                    if ([queue[j][i] isKindOfClass:[NSString class]]) { //If it's a string, could be a key or a string string
                        NSLog(@"%@",queue[j][i]);
                        if ([objects objectForKey:queue[j][i]]) {
                            [invocation setArgument:&(id){ objects[queue[j][i]] } atIndex:i-1];
                        } else if([queue[j][i] isEqualToString:@"error"] || [queue[j][i] isEqualToString:@"none"]) { //todo just use a dict?
                            NSError *error = nil;
                            [invocation setArgument:&error atIndex:i-1];
                        } else if([queue[j][i] isEqualToString:@"false"]) {
                            [invocation setArgument:&(BOOL){NO} atIndex:i-1];
                        } else if([queue[j][i] isEqualToString:@"true"]) {
                            [invocation setArgument:&(BOOL){YES} atIndex:i-1];
                        } else if([queue[j][i] isEqualToString:@"MTLIndirectCommandTypeConcurrentDispatch"]) {
                            [invocation setArgument:(&(MTLIndirectCommandType){MTLIndirectCommandTypeConcurrentDispatch}) atIndex:i-1];
                        } else if([queue[j][i] isEqualToString:@"MTLResourceCPUCacheModeDefaultCache"]) {
                            [invocation setArgument:(&(MTLResourceOptions){MTLResourceCPUCacheModeDefaultCache}) atIndex:i-1];
                        } else if([queue[j][i] isEqualToString:@"MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite"]) {
                            [invocation setArgument:(&(MTLResourceUsage){MTLResourceUsageRead | MTLResourceUsageWrite}) atIndex:i-1];
                        } else if(i == 3 && [queue[j][1] isEqualToString:@"setBytes:length:atIndex:"]) { // string to bytes
                            uint8_t *byteArgument = convertNSStringToBytes(queue[j][3]);
                            [invocation setArgument:&byteArgument atIndex:i-1];
                        } else {
                            [invocation setArgument:&(NSString *){queue[j][i]} atIndex:i-1]; //IF NOT IN OBJECTS
                        }
                    }
                }
                [invocation invoke];
                if ([queue[j] count] == 4+[queue[j][2] intValue]) {
                    __unsafe_unretained id result = nil;
                    [invocation getReturnValue:&result];
                    [objects setObject:result forKey:[queue[j] lastObject]];
                }
            }
        }
        [queue removeAllObjects];
        const char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
        send(handle, response, strlen(response), 0);
        close(handle);
    }
}

@end





