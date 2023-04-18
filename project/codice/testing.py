from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from lambeq import SpacyTokeniser
from discopy import Dim, grammar
from utilities import *

def find_faults_in_file(file: str):
    tokeniser = SpacyTokeniser()
    parser = BobcatParser(verbose = "progress")
    
    labels, sentences = extract_data(file)
    tokens = tokeniser.tokenise_sentences(sentences)

    faults = []
    i = 0
    count = 0
    while i < len(tokens):
        try:
            print(f"parsing string {i} of {len(tokens)}")
            diagram = parser.sentence2diagram(tokens[i], tokenised = True)
            i += 1
        except Exception: 
            faults.append(sentences[i])
            count += 1
            i += 1
            continue
        
    return faults
    
def check_fixed_faults(tokens_list):
    tokeniser = SpacyTokeniser()
    parser = BobcatParser(verbose = "progress")
    tokens = tokeniser.tokenise_sentences(tokens_list)

    for i in range(len(tokens)):
        try:
            print(f"parsing sentence {i} of {len(tokens)}")
            diagram = parser.sentence2diagram(tokens[i], tokenised=True)
        except Exception:
            print(f"Error on sentence {i}")
            continue

    print("Loop done")





fault_strings_cpn = [
    "The CNG should detect replayed user credentials and/or device credentials .",
    "When the CNG detects replayed user credentials and/or device credentials, the CNG shall stop the relevant processes .",
    "The CNG and the CPN shall be able to support parental control related functionalities limiting the use of the broadband connection on a user basis or time basis. Limitations on a content basis may be shared with devoted network servers .",
    "The CNG shall be equipped with a WAN interface towards the NGN, implementing layer 1 functionalities and layer 2 functionalities ('one-box' solution) .",
    "The CNG shall support different IP address schemes and subnets on the same physical LAN port and on different LAN ports, irrespective of routed mode or bridged mode of operation, allowing the direct addressability of CNDs from the NGN side in relation to data plane flows, control plane flows and management plane flows .",
    "The CNG and the Customer Network shall assure the confidentiality flows, the integrity of signalling flows control flows and media flows and management flows .",
    "The CNG and the Customer Network shall provide the opportunity for a customer network administrator to perform service configuration and network-related configuration. According to the service choices and network provider choices, the CNG and Customer Network may prevent user initiated modification of network parameters and service related parameters .",
    "The CNG shall support mechanisms for managing IPTV flows provided both in unicast mode and multicast mode .",
    "STB gateways or media gateways should be equipped with a programmable open API allowing the implementation of specific service logics .",
    "The CNG and the CPN shall support adequate Quality of Service mechanisms (e.g. prioritize the traffic in the event that there is not sufficient bandwidth in the customer network in order to ensure the correct network performances to each traffic flow). To achieve this objective a number of functionalities shall be supported by CNG and/or CNDs, as defined in the following bullets. Most of the following set of requirements is defined adopting as a reference the Home Gateway Initiative (HGI) Release Specification mentioned in clause 2 .",
    "In order to support the NGN services and intra-CPN communications, all the CNDs in the CPN shall be addressable directly or by the mean of the CNG using L2 mechanisms or L3 mechanisms .",
    "In case of managed services, the CNG shall support zero-touch provisioning to activate new services, starting from Internet access to voice and video services and shall be remotely manageable. In case of unmanaged services the user shall be able to configure the CNG by himself ."
]

fault_strings_epurse = [
    "On-line authentication must take place between the card issuer and the CEP card for load transactions, unload transactions and currency exchange transactions .",
    "The transaction signatures ensure end to end integrity of transmitted data for load transactions, unload transactions and currency exchange transactions .",
    "If a scheme provider establishes a central error repository, all transactions for the scheme with MAC errors must be sent to that central error repository even if they are submitted to the dispute process .",
    "Payment decisions are based on signature validations, scheme provider rules and merchant acquirer agreements .",
    "All load transactions are on-line transactions. Authorization of funds for load transactions must require a form of cardholder verification. The load device must support on-line encrypted PIN verification or off-line PIN verification .",  
    "Flexibility is required to accommodate the variety of environments where unlinked loads may be implemented. As a result, the design specification must not preclude dual-leg transactions from taking place either sequentially or in parallel. The design of a given implementation will vary depending on the device capabilities, host capabilities and network capabilities .",
    "Unload transactions and currency exchange transactions are optional for CEP card issuers. The CEP card must indicate whether the card issuer supports these transactions. However, if a card issuer issues multi-currency capable cards, it must provide its cardholder with a facility to remove any remaining value. As a result, if a card issuer supports loading of multiple currencies onto a card, then it must support the unload transaction or currency exchange transaction or both .",
    "Script messages that conform to EMV specifications may be included as part of load messages, unload messages and currency exchange messages from the card issuer to the CEP card. An update key must be used when card parameters are changed. Script messages may be sent to the CEP card either before or after the credit for load commands, debit of unload commands and currency exchange commands .",
    "It is a card issuer decision to determine the currencies that are allowed to occupy slots in the CEP card. This decision is made by the card issuer during the load transaction or currency exchange transaction, by approving the request or rejecting the request to authorize the transaction .",
    "Load functions and unload functions must be authenticated using end-to-end security between the card and the card issuer .",
    "The issuer host must authenticate the card upon the load request and unload request ."
]

check_fixed_faults(fault_strings_cpn)
#check_fixed_faults(fault_strings_epurse)


































"""
labels, sentences = extract_data("code/datasets/ePurse_edited.csv")
#tokens = tokeniser.tokenise_sentences(sentences)



faults = []
i = 0
count = 0
while i < len(sentences):
    try:
        print(f"parsing string {i} of {len(sentences)}")
        diagram = parser.sentence2diagram(sentences[i], tokenised = True)
        i += 1
    except Exception: 
        faults.append((labels[i], sentences[i]))
        count += 1
        i += 1
        continue
    """


