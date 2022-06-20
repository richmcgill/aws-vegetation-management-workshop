// code for this frontend is a modified version of https://github.com/aws-samples/s3uploader-ui

import React, {useState, useRef} from 'react';
import {useLocation, useNavigate} from 'react-router-dom';
import '@aws-amplify/ui-react/styles.css';
import './App.css';
import AppLayout from "@awsui/components-react/app-layout";
import FormField from "@awsui/components-react/form-field";
import Alert from "@awsui/components-react/alert";
import Container from "@awsui/components-react/container";
import Header from "@awsui/components-react/header";
import SideNavigation from '@awsui/components-react/side-navigation';
import Button from "@awsui/components-react/button";
import TokenGroup from "@awsui/components-react/token-group";
import TopNavigation from "@awsui/components-react/top-navigation"
import SpaceBetween from "@awsui/components-react/space-between";
import ProgressBar from "@awsui/components-react/progress-bar";
import Amplify, {Auth, Storage} from 'aws-amplify';
import {Authenticator} from '@aws-amplify/ui-react';
import axios from 'axios';
import awsconfig from './aws-exports';

Amplify.configure(awsconfig);

const api_invoke_url = 'REPLACE'

const appLayoutLabels = {
    navigation: 'Side navigation',
    navigationToggle: 'Open side navigation',
    navigationClose: 'Close side navigation',
    notifications: 'Notifications',
    tools: 'Help panel',
    toolsToggle: 'Open help panel',
    toolsClose: 'Close help panel'
};

const ServiceNavigation = () => {
    const location = useLocation();
    let navigate = useNavigate();

    function onFollowHandler(event) {
        if (!event.detail.external) {
            event.preventDefault();
            navigate(event.detail.href);
        }
    }

    return (
        <SideNavigation
            activeHref={location.pathname}
            header={{href: "/", text: "Vegetation Management"}}
            onFollow={onFollowHandler}
            items={[
                {type: "link", text: "Upload", href: "/"},
                {type: "divider"},
                {
                    type: "link",
                    text: "AWS Solutions Architect",
                    href: "https://workshops.aws",
                    external: true
                }
            ]}
        />
    );
}

function formatBytes(a, b = 2, k = 1024) {
    let d = Math.floor(Math.log(a) / Math.log(k));
    return 0 === a ? "0 Bytes" : parseFloat((a / Math.pow(k, d)).toFixed(Math.max(0, b))) + " " + ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"][d];
}

const Content = () => {
    const hiddenFileInput = useRef(null);
    const [visibleAlert, setVisibleAlert] = useState(false);
    const [uploadList, setUploadList] = useState([]);
    const [fileList, setFileList] = useState([]);
    const [historyList, setHistoryList] = useState([]);
    const [historyCount, setHistoryCount] = useState(0);
    const handleClick = () => {
        hiddenFileInput.current.value = ""; // This avoids errors when selecting the same files multiple times
        hiddenFileInput.current.click();
    };
    const handleChange = e => {
        e.preventDefault();
        let i, tempUploadList = [];
        for (i = 0; i < e.target.files.length; i++) {
            tempUploadList.push({
                label: e.target.files[i].name,
                labelTag: formatBytes(e.target.files[i].size),
                description: 'File type: ' + e.target.files[i].type,
                icon: 'file',
                id: i
            })
        }
        setUploadList(tempUploadList);
        setFileList(e.target.files);
    };

    function progressBarFactory(fileObject) {
        let localHistory = historyList;
        const id = localHistory.length;
        localHistory.push({
            id: id,
            percentage: 0,
            filename: fileObject.name,
            filetype: fileObject.type,
            filesize: formatBytes(fileObject.size),
            status: 'in-progress',
            data: null
        });
        setHistoryList(localHistory);
        return (progress) => {
            let tempHistory = historyList.slice();
            const percentage = Math.round((progress.loaded / progress.total) * 100);
            tempHistory[id].percentage = percentage;
            if (percentage === 100) {
                tempHistory[id]['status'] = 'success';
                handleApiCall(tempHistory[id]['id'], fileObject.name);
            }
            setHistoryList(tempHistory);
        };
    }
    
    async function handleApiCall(id, name) {
        const api = api_invoke_url;
        const data = { "body" : name };
        let tempHistory = historyList.slice();
        await axios.post(api, data).then((response) => { 
            const api_data = 'data:image/png;base64,' + JSON.parse(JSON.stringify(response))["data"];
            tempHistory[id]['data'] = api_data
            setHistoryList(tempHistory);
        }).catch((error) => {
          console.log(error.response.data);
        });
    }
  

    const handleUpload = () => {
        if (uploadList.length === 0) {
            setVisibleAlert(true);
        } else {
            let i, progressBar = [], uploadCompleted = [];
            for (i = 0; i < uploadList.length; i++) {
                // If the user has removed some items from the Upload list, we need to correctly reference the file
                const id = uploadList[i].id;
                progressBar.push(progressBarFactory(fileList[id]));
                setHistoryCount(historyCount + 1);
                uploadCompleted.push(Storage.put(fileList[id].name, fileList[id], {
                        progressCallback: progressBar[i],
                        level: "public"
                    }).then(result => {
                        // Trying to remove items from the upload list as they complete. Maybe not work correctly
                        // setUploadList(uploadList.filter(item => item.label !== result.key));
                        console.log(`Completed the upload of ${result.key}`);
                    })
                );
            }
            // When you finish the loop, all items should be removed from the upload list
            Promise.all(uploadCompleted)
                .then(() => setUploadList([]));
        }
    }

    const handleDismiss = (itemIndex) => {
        setUploadList([
            ...uploadList.slice(0, itemIndex),
            ...uploadList.slice(itemIndex + 1)
        ]);
    };

    const List = ({list}) => (
        <>
            {list.map((item) => (
                <>
                    <ProgressBar
                        key={item.id}
                        status={item.status}
                        value={item.percentage}
                        variant="standalone"
                        additionalInfo={item.filesize}
                        description={item.filetype}
                        label={item.filename}
                    />
                    {(item.data == null) ? 'Waiting for model response...' : <img src = {item.data} width="750" height="250"/> }
                </>
            ))}
        </>
    );
    return (
        <SpaceBetween size="l">
            <Container
                id="model-upload-image"
                header={
                    <Header variant="h2">
                        Upload image to SageMaker model
                    </Header>
                }
            >
                {
                    <div>
                        <Alert
                            onDismiss={() => setVisibleAlert(false)}
                            visible={visibleAlert}
                            dismissAriaLabel="Close alert"
                            dismissible
                            type="error"
                            header="No files selected"
                        >
                            You must select the files that you want to upload.
                        </Alert>

                        <FormField
                            label='Image Upload'
                            description='Click on the Open button and select the image that you want to upload'
                        />

                        <SpaceBetween direction="horizontal" size="xs">
                            <Button onClick={handleClick}
                                    iconAlign="left"
                                    iconName="upload"
                            >
                                Choose image
                            </Button>
                            <input
                                type="file"
                                ref={hiddenFileInput}
                                onChange={handleChange}
                                style={{display: 'none'}}
                                accept=".tiff,.tif"
                            />
                            <Button variant="primary" onClick={handleUpload}>Upload</Button>
                        </SpaceBetween>

                        <TokenGroup
                            onDismiss={({detail: {itemIndex}}) => {
                                handleDismiss(itemIndex)
                            }}
                            items={uploadList}
                            alignment="vertical"
                            limit={1}
                        />
                    </div>
                }
            </Container>
            <Container
                id="history"
                header={
                    <Header variant="h2">
                        History
                    </Header>
                }
            >
                <List list={historyList}/>
            </Container>
        </SpaceBetween>

    );
};

function App() {
    const [navigationOpen, setNavigationOpen] = useState(true);
    const navbarItemClick = e => {
        console.log(e);
        if (e.detail.id === 'signout') {
            Auth.signOut().then(() => {
                window.location.reload();
            });
        }
    }

    return (
        <Authenticator>
            {({signOut, user}) => (
                <>
                    <div id="navbar" style={{fontSize: 'body-l !important', position: 'sticky', top: 0, zIndex: 1002}}>
                        <TopNavigation
                            identity={{
                                href: "#",
                                title: "Vegetation Management Workshop",
                                logo: {
                                    src: "data:image/png;base64,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",
                                    alt: "Vegetation Management Workshop"
                                }
                            }}
                            utilities={[
                                {
                                    type: "button",
                                    text: "AWS",
                                    href: "https://aws.amazon.com/",
                                    external: true,
                                    externalIconAriaLabel: " (opens in a new tab)"
                                },
                                {
                                    type: "menu-dropdown",
                                    text: user.attributes.email,
                                    description: user.attributes.email,
                                    iconName: "user-profile",
                                    onItemClick: navbarItemClick,
                                    items: [
                                        {id: "profile", text: "Profile"},
                                        {id: "preferences", text: "Preferences"},
                                        {id: "security", text: "Security"},
                                        {
                                            id: "feedback",
                                            text: "Feedback",
                                            href: "#",
                                            external: true,
                                            externalIconAriaLabel:
                                                " (opens in new tab)"
                                        },
                                        {id: "signout", text: "Sign out"}
                                    ]
                                }
                            ]}
                            i18nStrings={{
                                searchIconAriaLabel: "Search",
                                searchDismissIconAriaLabel: "Close search",
                                overflowMenuTriggerText: "More"
                            }}
                        />
                    </div>
                    <AppLayout
                        content={<Content/>}
                        headerSelector='#navbar'
                        navigation={<ServiceNavigation/>}
                        navigationOpen={navigationOpen}
                        onNavigationChange={({detail}) => setNavigationOpen(detail.open)}
                        ariaLabels={appLayoutLabels}
                    />
                </>
            )}
        </Authenticator>
    );
}

export default App;