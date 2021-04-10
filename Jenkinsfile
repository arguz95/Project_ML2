node {

    checkout scm

    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub') {

        def customImage = docker.build("ram1633/bankruptcy")

        /* Push the container to the custom Registry */
        customImage.push()
    }
}
